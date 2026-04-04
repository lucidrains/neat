import std/[
  random,
  times,
  strformat,
  options,
  assertions,
  math,
  sequtils,
  sets,
  setutils,
  tables,
  sugar,
  segfaults,
  atomics,
  algorithm,
  os,
  json
]

import nimpy
import nimpy/[raw_buffers, py_types]

import arraymancer
import bitvector
import jsony
import malebolgia

var master = create_master()

type
  Prob = range[0.0..1.0]

# init

randomize()

# templates

template benchmark(
  name: string,
  trials: int,
  code: untyped
) =
  var result = 0.0

  for _ in 0..<trials:
    let start_time = epoch_time()
    code
    let diff = epoch_time() - start_time
    result += diff / trials

  echo name & ": average time over " & $trials & " trials is " & $(result * 1e3) & " ms"

# from @Vindaar - https://github.com/yglukhov/nimpy/issues/114#issuecomment-531504502

type
  NumpyArray[T] = object
    py_buf: ptr RawPyBuffer
    data: ptr UncheckedArray[T]
    shape: seq[int]
    len: int

proc init_nd_array[T](ar: PyObject): NumpyArray[T] =
  result.py_buf = cast[ptr RawPyBuffer](alloc0(sizeof(RawPyBuffer)))
  ar.get_buffer(result.py_buf[], PyBUF_WRITABLE or PyBUF_ND)
  let shapear = cast[ptr UncheckedArray[Py_ssize_t]](result.py_buf.shape)

  for i in 0 ..< result.py_buf.ndim:
    let dimsize = shapear[i].int # py_ssize_t is csize
    result.shape.add dimsize

  result.len = result.shape.foldl(a * b, 1)
  result.data = cast[ptr UncheckedArray[T]](result.py_buf.buf)

proc release[T](nd: var NumpyArray[T]) =
  nd.py_buf[].release()
  dealloc(nd.py_buf)

template parse_indices(
  indices: seq[int],
  shape: seq[int]
): int =
  var
    res = 0
    stride = 1

  for i in countdown(indices.high, 0):
    res += indices[i] * stride
    stride *= shape[i]

  res

# functions

proc satisfy_prob(prob: Prob): bool =
  if prob == 0.0: return false
  if prob == 1.0: return true
  rand(1.0) < prob

proc coin_flip(): bool = satisfy_prob(0.5)

proc random_normal(eps: float = 1e-30): float32 =
  sqrt(-2 * ln(max(eps, rand(1.0)))) * cos(2 * PI * rand(1.0))

# fast genetic algorithm - https://arxiv.org/abs/1703.03334

proc sample_power_law(half_len: int, beta: float): int =
  ## sample k from discrete power law P(k) ∝ k^{-β}, k ∈ [1, half_len]
  var total = 0.0
  for k in 1..half_len:
    total += pow(k.float, -beta)
  let r = rand(total)
  var cumsum = 0.0
  for k in 1..half_len:
    cumsum += pow(k.float, -beta)
    if r <= cumsum: return k
  half_len

proc sample_waiting_time(prob: float): int =
  ## geometric skip - number of trials until first success
  if prob <= 0.0: return int.high
  if prob >= 1.0: return 1
  1 + (ln(1.0 - max(rand(1.0), 1e-15)) / ln(1.0 - prob)).floor.int

# activation functions

proc sigmoid(x: float32): float32 = 1.0 / (1.0 + exp(-x))
proc relu(x: float32): float32 = max(x, 0.0)
proc gauss(x: float32): float32 = exp(-pow(x, 2))
proc identity(x: float32): float32 = x
proc elu(x: float32): float32 = (if x >= 0.0: x else: exp(x) - 1.0)
proc clamp_one(x: float32): float32 = max(min(x, 1.0), -1.0)

type
  Activation = enum
    identity, sigmoid, tanh, relu, clamp, elu, gauss, sin, abs

  NumNodesInfo = tuple
    num_inputs: int
    num_outputs: int
    num_nodes: int

  WeightUpdate = tuple
    weight: float32
    from_node_id: int

  NodeUpdate = tuple
    to_node_id: int
    activation_id: int
    bias: float32
    trace: seq[WeightUpdate]

  ExecTrace = ref object
    node_info: NumNodesInfo
    node_updates: seq[NodeUpdate]

  NodeType = enum
    input, output, hidden

  Node = object
    id: int
    topology_id: int
    `type`: NodeType = hidden

  Edge = object
    id: int
    topology_id: int
    from_node_id: int
    to_node_id: int

  MetaNode = ref object
    topology_id: int
    node_id: int
    disabled: bool
    can_disable: bool
    can_change_activation: bool = true
    activation: Activation = tanh
    bias: float32 = 0.0

  MetaEdge = ref object
    topology_id: int
    edge_id: int
    can_disable: bool = true
    disabled: bool
    weight: float32
    local_from_node_id: int
    local_to_node_id: int

  NeuralNetwork = ref object
    id: int
    topology_id: int
    num_inputs: int
    num_outputs: int
    num_hiddens: seq[int] = @[]
    meta_nodes: seq[MetaNode] = @[]
    meta_edges: seq[MetaEdge] = @[]
    cached_exec_trace: Option[ExecTrace]

  # hyperparams

  SelectionHyperParams = object
    frac_natural_selected: float = 0.5
    tournament_size: int = 3
    use_queen_bee: bool = false
    queen_strong_mutation_rate: float = 0.25

  CrossoverHyperParams = object
    prob_child_disabled_given_parent_cond: float = 0.75
    prob_remove_disabled_node: float = 0.01
    prob_inherit_all_excess_genes: float = 1.0

  MutationHyperParams = object
    mutate_prob: float = 0.95
    use_fast_ga: bool = false
    fast_ga_beta: float = 1.5
    add_novel_edge_prob: float = 5e-3
    toggle_meta_edge_prob: float = 0.05
    add_remove_node_prob: float = 1e-5
    change_activation_prob: float = 0.001
    change_edge_weight_prob: float = 0.5
    replace_edge_weight_prob: float = 0.1
    change_node_bias_prob: float = 0.1
    replace_node_bias_prob: float = 0.1
    grow_edge_prob: float = 5e-4
    grow_node_prob: float = 1e-5
    perturb_weight_strength: float = 0.1
    perturb_bias_strength: float = 0.1
    num_preserve_elites: int = 0
    max_weight_magnitude: float = 30.0

  # 'topology' - rename to population at some point

  Topology = ref object
    id: int

    nodes: seq[Node] = @[] # nodes will be always arrange [input] [output] [hiddens]
    edges: seq[Edge] = @[]

    nodes_index: Table[int, Node]
    edges_index: Table[int, Edge]
    conn_index: Table[(int, int), int]
    edge_splits: Table[int, (int, int, int)]

    num_inputs: int
    num_outputs: int
    num_hiddens: seq[int] = @[]

    nn_id: int = 0
    node_innovation_id: Atomic[int]
    edge_innovation_id: Atomic[int]

    num_islands: int = 1
    pop_size: int = 0
    curr_pop_size: int = 0
    population: seq[NeuralNetwork] = @[]

    # default hyperparams

    selection_hyper_params: SelectionHyperParams

    crossover_hyper_params: CrossoverHyperParams

    mutation_hyper_params: MutationHyperParams

  # crossover related

  ParentAndFitness* = tuple
    index: int
    fitness: float32

  Couple* = tuple
    parent1: ParentAndFitness
    parent2: ParentAndFitness

  Couples* = seq[Couple]

  TopologyInfo = object
    total_innovated_nodes: int
    total_innovated_edges: int

# globals

var topologies = init_table[int, Topology]()
var topology_id: Atomic[int]

# helper accessors

proc get_topology_info(
  top_id: int
): TopologyInfo {.exportpy.} =

  let top = topologies[top_id]

  result.total_innovated_nodes = top.node_innovation_id.load + 1
  result.total_innovated_edges = top.edge_innovation_id.load + 1

# saving population to pretty json for introspecting on evolved graphs

proc skip_hook*(T: typedesc[NeuralNetwork], key: static string): bool =
  key in ["cached_exec_trace"]

proc skip_hook*(T: typedesc[Topology], key: static string): bool =
  key in ["conn_index", "edges_index", "nodes_index", "node_innovation_id", "edge_innovation_id"]

proc save_json_to_file(
  top_id: int,
  filepath: string
) {.exportpy.} =

  let top = topologies[top_id]
  let contents = (top.nodes, top.edges, top.population).to_json()

  let (dir, name, ext) = split_file(filepath)
  if not dir_exists(dir):
    create_dir(dir)

  write_file(filepath, contents.parse_json().pretty())

# functions

proc add_node(top: Topology, node_type: NodeType = hidden): int
proc add_edge(top: Topology, from_node_id: int, to_node_id: int): int

proc activate(act: Activation, input: Tensor[float32]): Tensor[float32] {.gcsafe.}
proc activate(act: Activation, input: float32): float32 {.gcsafe.}
proc activate(node: MetaNode, input: Tensor[float32]): Tensor[float32] {.gcsafe.}

proc set_population_exec_trace(top_id: int)

proc rand_activation(): Activation

proc add_topology(
  num_inputs: int,
  num_outputs: int,
  num_hiddens: seq[int],
  mutation_hyper_params: MutationHyperParams = MutationHyperParams(),
  crossover_hyper_params: CrossoverHyperParams = CrossoverHyperParams(),
  selection_hyper_params: SelectionHyperParams = SelectionHyperParams(),
  num_islands: int = 1
): int {.exportpy.} =

  let topology = Topology(
    id: topology_id.fetch_add(1),
    num_inputs: num_inputs,
    num_outputs: num_outputs,
    num_hiddens: num_hiddens,
    mutation_hyper_params: mutation_hyper_params,
    crossover_hyper_params: crossover_hyper_params,
    selection_hyper_params: selection_hyper_params,
    num_islands: num_islands
  )

  topology.nodes_index = init_table[int, Node]()
  topology.edges_index = init_table[int, Edge]()
  topology.conn_index = init_table[(int, int), int]()
  topology.edge_splits = init_table[int, (int, int, int)]()

  topologies[topology.id] = topology

  # create input and output nodes

  let
    input_node_ids = (0..<num_inputs).to_seq
    output_node_ids = (num_inputs..<(num_inputs + num_outputs)).to_seq

  for _ in 0..<num_inputs:
    discard add_node(topology, NodeType.input)

  for _ in 0..<num_outputs:
    discard add_node(topology, NodeType.output)

  # create edges

  for input_id in input_node_ids:
    for output_id in output_node_ids:
      discard add_edge(topology, input_id, output_id)

  # initial pool of hidden nodes and edges, all disabled for new neural networks at start

  var hidden_node_ids: seq[int] = @[]

  for num_hidden_layer in num_hiddens:
    var layer_hidden_ids: seq[int] = @[]

    for _ in 0..<num_hidden_layer:
      layer_hidden_ids.add(add_node(topology, NodeType.hidden))

    hidden_node_ids.add(layer_hidden_ids)

  var all_ids: seq[seq[int]] = @[input_node_ids] & hidden_node_ids & @[output_node_ids]

  for layer_index, from_layer_ids in all_ids[0..^2]:

    let to_layer_ids = all_ids[layer_index + 1]

    for from_id in from_layer_ids:
      for to_id in to_layer_ids:
        discard add_edge(topology, from_id, to_id)

  # return id

  return topology.id

proc remove_topology(topology_id: int) {.exportpy.} =
  topologies.del(topology_id)

proc add_node(
  top: Topology,
  node_type: NodeType = hidden
): int {.exportpy.} =

  # create node, increment primary key, and add to global nodes

  let node = Node(
    id: top.node_innovation_id.fetch_add(1),
    type: node_type,
    topology_id: top.id
  )

  top.nodes.add(node)
  top.nodes_index[node.id] = node

  return node.id

proc add_edge(
  top: Topology,
  from_node_id: int,
  to_node_id: int
): int {.exportpy.} =

  # validate node id

  let max_node_id = top.nodes.len
  assert(0 <= from_node_id  and from_node_id < max_node_id)
  assert(0 <= to_node_id and to_node_id < max_node_id)

  # create edge, increment primary key and add to global edges

  let edge = Edge(
    id: top.edge_innovation_id.fetch_add(1),
    topology_id: top.id,
    from_node_id: from_node_id,
    to_node_id: to_node_id
  )

  top.edges.add(edge)
  top.edges_index[edge.id] = edge
  top.conn_index[(from_node_id, to_node_id)] = edge.id

  return edge.id

# main evolutionary functions

# population functions

proc init_nn(
  top_id: int,
  nn_id: int,
  sparsity: float32 = 0.05
) =
  let top = topologies[top_id]

  let nn = NeuralNetwork(
    num_inputs: top.num_inputs,
    num_outputs: top.num_outputs,
    num_hiddens: top.num_hiddens
  )

  let
    output_node_index_start = top.num_inputs
    hidden_node_index_start = top.num_inputs + top.num_outputs
    hidden_edge_index_start = top.num_inputs * top.num_outputs

  # index from global node id to local node id

  var node_index = init_table[int, int]()

  # initialize input nodes

  for node in top.nodes[0..<top.num_inputs]:

    let meta_node = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: false,
      can_disable: false,
      can_change_activation: false,
      activation: identity
    )

    node_index[node.id] = nn.meta_nodes.len
    nn.meta_nodes.add(meta_node)

  # initial output nodes
  
  for node in top.nodes[output_node_index_start..<(output_node_index_start + top.num_outputs)]:

    let meta_node = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: false,
      can_disable: false,
      can_change_activation: false,
      activation: identity
    )

    node_index[node.id] = nn.meta_nodes.len
    nn.meta_nodes.add(meta_node)

  # hiddens

  for node in top.nodes[hidden_node_index_start..<top.nodes.len]:

    let meta_node = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: satisfy_prob(sparsity),
      can_disable: true,
      bias: random_normal(),
      can_change_activation: false,
      activation: if coin_flip(): relu else: sigmoid
    )

    node_index[node.id] = nn.meta_nodes.len
    nn.meta_nodes.add(meta_node)

  # create edges - start off with only fully connected from inputs to outputs

  for index, edge in top.edges[0..<(top.num_inputs * top.num_outputs)]:

      let meta_edge = MetaEdge(
        topology_id: top.id,
        edge_id: edge.id,
        disabled: false,
        can_disable: coin_flip(),
        local_from_node_id: node_index[edge.from_node_id],
        local_to_node_id: node_index[edge.to_node_id],
        weight: random_normal()
      )

      nn.meta_edges.add(meta_edge)

  for edge in top.edges[hidden_edge_index_start..<top.edges.len]:

    let meta_edge = MetaEdge(
      topology_id: top.id,
      edge_id: edge.id,
      disabled: satisfy_prob(sparsity),
      local_from_node_id: node_index[edge.from_node_id],
      local_to_node_id: node_index[edge.to_node_id],
      weight: random_normal()
    )

    nn.meta_edges.add(meta_edge)

  # add neural net to population

  top.population[nn_id] = nn

proc init_population(
  top_id: int,
  pop_size: range[1..int.high],
) {.exportpy.} =

  let top = topologies[top_id]
  assert top.pop_size == 0

  top.pop_size = pop_size
  top.curr_pop_size = pop_size
  top.population = new_seq[NeuralNetwork](pop_size)

  for nn_id in 0 ..< pop_size:
    init_nn(top_id, nn_id)

  assert top.population.len == top.pop_size

  set_population_exec_trace(top_id)

# forward

proc evaluate_nn(
  top_id: int,
  nn_id: int,
  seq_inputs: seq[seq[float32]]
): seq[seq[float32]] {.exportpy.} =

  let inputs = seq_inputs.map(seq_float32 => seq_float32.to_tensor)

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  assert top.num_inputs == inputs.len

  let one_input = inputs[0]
  let one_input_shape = one_input.shape

  let num_nodes = nn.meta_nodes.len

  var visited = new_bit_vector[uint](num_nodes)
  var finished = new_bit_vector[uint](num_nodes)
  var values = new_seq[Tensor[float32]](num_nodes)

  # global to local node indices

  var node_index = init_table[int, int]()

  # global edge index to local node from and to index

  var edge_index = init_table[int, (int, int)]()

  for local_node_index, meta_node in nn.meta_nodes:

    if meta_node.disabled:
      continue

    node_index[meta_node.node_id] = local_node_index

  for edge in top.edges:
    if node_index.has_key(edge.from_node_id) and node_index.has_key(edge.to_node_id):

      edge_index[edge.id] = (
        node_index[edge.from_node_id],
        node_index[edge.to_node_id],
      )

  # set the values of inputs

  for i in 0..<nn.num_inputs:
    finished[i] = 1

    values[i] = nn.meta_nodes[i].activate(inputs[i])

  # proc for fetching value of node at a given meta node index

  proc compute_node_value(
    index: int,
    visited: BitVector
  ): Tensor[float32] =

    if finished[index] == 1:
      return values[index]

    let meta_node = nn.meta_nodes[index]

    # start with bias

    var next_visited = visited
    next_visited[index] = 1

    var node_value = zeros[float32](one_input_shape) +. meta_node.bias

    # find all edges

    var input_node_index_and_weight: seq[(float32, int)] = @[] # omit visited

    for meta_edge in nn.meta_edges:
      if meta_edge.disabled:
        continue

      if not edge_index.has_key(meta_edge.edge_id):
        # node may be disabled above
        continue

      let (local_from_node_id, local_to_node_id) = edge_index[meta_edge.edge_id]

      if local_to_node_id != index:
        continue

      let meta_node = nn.meta_nodes[local_from_node_id]

      if next_visited[local_from_node_id] == 1:
        continue

      if meta_node.disabled:
        continue

      input_node_index_and_weight.add((meta_edge.weight, local_from_node_id))

    # get weighted sum of inputs to the node

    for entry in input_node_index_and_weight:
      let (weight, local_from_node_id) = entry
      node_value += weight * compute_node_value(local_from_node_id, next_visited)

    # activation

    let activated_value = meta_node.activate(node_value)

    finished[index] = 1
    values[index] = activated_value

    return activated_value

  # compute outputs

  for i in 0..<nn.num_outputs:
    let output_index = nn.num_inputs + i
    let output_value = compute_node_value(output_index, visited)

    result.add(output_value.to_seq)

# caching graph execution trace

proc evaluate_nn_exec_trace(
  top: Topology,
  nn_id: int,
): ExecTrace {.exportpy.} =
  var trace = new_seq[NodeUpdate]()

  let nn = top.population[nn_id]

  let num_nodes = nn.meta_nodes.len

  var visited = new_bit_vector[uint](num_nodes)
  var finished = new_bit_vector[uint](num_nodes)

  # global to local node indices

  var node_index = init_table[int, int]()

  # global edge index to local node from and to index

  var edge_index = init_table[int, (int, int)]()

  for local_node_index, meta_node in nn.meta_nodes:

    if meta_node.disabled:
      continue

    node_index[meta_node.node_id] = local_node_index

  for edge in top.edges:
    if node_index.has_key(edge.from_node_id) and node_index.has_key(edge.to_node_id):

      edge_index[edge.id] = (
        node_index[edge.from_node_id],
        node_index[edge.to_node_id],
      )

  # set the values of inputs

  for i in 0..<nn.num_inputs:
    let node_index = i + num_nodes # external inputs placed at the end
    let input_node = nn.meta_nodes[i]

    finished[i] = 1

    trace.add((
      i,
      input_node.activation.ord,
      0.0'f32,
      @[(1.0'f32, node_index)]
    ))

  # proc for fetching value of node at a given meta node index

  proc compute_node_value(
    index: int,
    visited: BitVector,
    trace: var seq[NodeUpdate]
  ) =

    if finished[index] == 1:
      return

    let meta_node = nn.meta_nodes[index]

    # start with bias

    var next_visited = visited
    next_visited[index] = 1

    # find all edges

    var input_node_index_and_weight: seq[(float32, int)] = @[] # omit visited

    for meta_edge in nn.meta_edges:
      if meta_edge.disabled:
        continue

      if not edge_index.has_key(meta_edge.edge_id):
        # node may be disabled above
        continue

      let (local_from_node_id, local_to_node_id) = edge_index[meta_edge.edge_id]

      if local_to_node_id != index:
        continue

      let meta_node = nn.meta_nodes[local_from_node_id]

      if next_visited[local_from_node_id] == 1:
        continue

      if meta_node.disabled:
        continue

      input_node_index_and_weight.add((meta_edge.weight, local_from_node_id))

    # get weighted sum of inputs to the node

    var multiplies: seq[(float32, int)] = @[]

    for entry in input_node_index_and_weight:
      let (weight, local_from_node_id) = entry
      compute_node_value(local_from_node_id, next_visited, trace)
      multiplies.add((weight, local_from_node_id))

    # activation

    finished[index] = 1

    trace.add((
      index,
      meta_node.activation.ord,
      meta_node.bias,
      multiplies
    ))

  # compute outputs

  for i in 0..<nn.num_outputs:
    let output_index = nn.num_inputs + i
    compute_node_value(output_index, visited, trace)

  return ExecTrace(
    node_info: (top.num_inputs, top.num_outputs, num_nodes),
    node_updates: trace
  )

proc evaluate_nn_exec_trace(
  top_id: int,
  nn_id: int,
): ExecTrace {.exportpy.} =

  let top = topologies[top_id]
  return evaluate_nn_exec_trace(top, nn_id)  

proc evaluate_nn_with_trace(
  trace_with_meta_info: ExecTrace,
  inputs: seq[seq[float32]]
): seq[seq[float32]] {.gcsafe exportpy.} =

  var meta_info = trace_with_meta_info.node_info
  var trace = trace_with_meta_info.node_updates
  let (num_inputs, num_outputs, num_nodes) = meta_info

  let seq_inputs_tensor = inputs.map(seq_float32 => seq_float32.to_tensor)

  let one_input = seq_inputs_tensor[0]
  let one_input_shape = one_input.shape

  var values = new_seq_with(num_nodes, zeros[float32](one_input_shape))

  values &= seq_inputs_tensor

  for update_node in trace:
    let (to_id, act_index, bias, incoming_weights) = update_node

    values[to_id] = values[to_id] +. bias

    for incoming_weight in incoming_weights:
      let (weight, from_id) = incoming_weight
      values[to_id] += weight * values[from_id]

    values[to_id] = Activation(act_index).activate(values[to_id])

  let seq_outputs = values[num_inputs ..< (num_inputs + num_outputs)]

  return seq_outputs.map(tensor => tensor.to_seq)

proc evaluate_nn_single(
  top_id: int,
  nn_id: int,
  inputs: seq[float32],
  use_exec_cache: bool = false
): seq[float32] {.exportpy.} =

  let seq_inputs = inputs.map(value => @[value])

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  if use_exec_cache and nn.cached_exec_trace.is_none:
    nn.cached_exec_trace = evaluate_nn_exec_trace(top_id, nn_id).some

  let outputs = if use_exec_cache:
    evaluate_nn_with_trace(nn.cached_exec_trace.get, seq_inputs)
  else:
    evaluate_nn(top_id, nn_id, seq_inputs)

  return outputs.map(tensor => tensor[0])

var thread_local_values {.threadvar.}: seq[float32]

proc evaluate_nn_single_with_trace_thread_fn(
  trace_ptr: pointer,
  buffer_input: ptr UncheckedArray[float32],
  buffer_output: ptr UncheckedArray[float32]
) {.gcsafe.} =

  let trace = cast[ExecTrace](trace_ptr)
  let num_inputs = trace.node_info.num_inputs
  var meta_info = trace.node_info
  var trace_updates = trace.node_updates
  let (_, num_outputs, num_nodes) = meta_info

  let req_len = num_nodes + num_inputs
  if thread_local_values.len < req_len:
    thread_local_values.setLen(req_len)

  # copy inputs to the end of the buffer
  for i in 0 ..< num_inputs:
    thread_local_values[num_nodes + i] = buffer_input[i]

  for update_node in trace_updates:
    let (to_id, act_index, bias, incoming_weights) = update_node

    var current_val = bias

    for incoming_weight in incoming_weights:
      let (weight, from_id) = incoming_weight
      current_val += weight * thread_local_values[from_id]

    thread_local_values[to_id] = Activation(act_index).activate(current_val)

  # write output
  for i in 0 ..< num_outputs:
    buffer_output[i] = thread_local_values[num_inputs + i]


proc set_population_exec_trace(
  top_id: int
) =
  let top = topologies[top_id]

  for nn_id in 0 ..< top.population.len:
    let nn = top.population[nn_id]

    # set the cached graph execution on neural network if not exists (it has been mutated)
    if nn.cached_exec_trace.is_none:
      nn.cached_exec_trace = evaluate_nn_exec_trace(top.id, nn_id).some

proc evaluate_population(
  top_id: int,
  inputs: PyObject,
  outputs: PyObject
) {.exportpy.} =

  let top = topologies[top_id]

  var nd_array_inputs = init_nd_array[float32](inputs)
  var nd_array_outputs = init_nd_array[float32](outputs)

  # validate expected shapes
  assert nd_array_inputs.shape.len == 2
  assert nd_array_outputs.shape.len == 2
  assert nd_array_inputs.shape[0] == top.pop_size
  assert nd_array_inputs.shape[1] == top.num_inputs
  assert nd_array_outputs.shape[0] == top.pop_size
  assert nd_array_outputs.shape[1] == top.num_outputs

  let num_inputs = top.num_inputs
  let num_outputs = top.num_outputs

  master.await_all:
    for nn_id in 0 ..< top.pop_size:
      let nn = top.population[nn_id]

      let buffer_input = cast[ptr UncheckedArray[float32]](nd_array_inputs.data[nn_id * num_inputs].addr)
      let buffer_output = cast[ptr UncheckedArray[float32]](nd_array_outputs.data[nn_id * num_outputs].addr)

      master.spawn evaluate_nn_single_with_trace_thread_fn(
        cast[pointer](nn.cached_exec_trace.get),
        buffer_input,
        buffer_output
      )

  nd_array_inputs.release()
  nd_array_outputs.release()

proc activate(
  node: MetaNode,
  input: Tensor[float32]
): Tensor[float32] {.gcsafe.} =

  return activate(node.activation, input)

proc activate(
  act: Activation,
  input: Tensor[float32]
): Tensor[float32]  {.gcsafe.} =

  return input.map(value => activate(act, value))

proc activate(
  act: Activation,
  input: float32
): float32 {.gcsafe.} =

  if act == identity:
    return input
  elif act == sigmoid:
    return sigmoid(input)
  elif act == tanh:
    return tanh(input)
  elif act == relu:
    return relu(input)
  elif act == clamp:
    return clamp_one(input)
  elif act == elu:
    return elu(input)
  elif act == gauss:
    return gauss(input)
  elif act == sin:
    return sin(input)
  elif act == abs:
    return abs(input)

proc rand_activation(): Activation =
  let activations = Activation.to_seq
  let rand_activation_index = rand(Activation.high.ord)
  result = activations[rand_activation_index]

# mutation and crossover

proc tournament*(
  fitnesses: seq[float32],
  num_tournaments: range[1..int.high],
  tournament_size: range[2..int.high]
): Couples {.exportpy.} =

  var gene_ids = arange(fitnesses.len).to_seq()

  for _ in 0..<num_tournaments:

    var
      parent1, parent2: int = -1
      fitness1, fitness2: float32 = -1e6

    shuffle(gene_ids)
    let tournament = gene_ids[0..<tournament_size]

    for gene_id in tournament:
      let gene_fitness = fitnesses[gene_id]

      if gene_fitness > fitness1:
        parent2 = parent1
        fitness2 = fitness1
        parent1 = gene_id
        fitness1 = gene_fitness

      elif gene_fitness > fitness2:
        parent2 = gene_id
        fitness2 = gene_fitness

    result.add(((parent1, fitness1), (parent2, fitness2)))

proc select_and_tournament(
  top_ids: seq[int],
  fitnesses: seq[float32],
  selection_hyper_params: Option[SelectionHyperParams] = SelectionHyperParams.none
): (
  seq[int],
  seq[float32],
  Couples,
  seq[int]
) {.exportpy.} =

  assert top_ids.len > 0
  let top = topologies[top_ids[0]]

  let num_islands = top.num_islands
  let pop_size = top.population.len
  assert pop_size mod num_islands == 0
  let island_pop_size = pop_size div num_islands

  let hyper_params = selection_hyper_params.get(top.selection_hyper_params)
  let num_selected_per_island = max(2, (hyper_params.frac_natural_selected * island_pop_size.float).int)
  let tournament_size = min(num_selected_per_island, hyper_params.tournament_size)

  assert tournament_size <= num_selected_per_island
  assert island_pop_size > num_selected_per_island

  var all_selected_indices: seq[int] = @[]
  var all_selected_fitnesses: seq[float32] = @[]
  var all_parent_indices: Couples = @[]
  var all_target_nn_ids: seq[int] = @[]

  for island_id in 0 ..< num_islands:
    let offset = island_id * island_pop_size
    let island_fitnesses = fitnesses[offset ..< offset + island_pop_size]

    let sorted_indices = island_fitnesses
      .to_tensor()
      .argsort(order = SortOrder.Descending)
      .to_flat_seq()

    let selected_sorted_indices = sorted_indices[0..<num_selected_per_island]
    let selected_sorted_fitnesses = selected_sorted_indices.map(index => island_fitnesses[index])

    let num_tournaments = island_pop_size - num_selected_per_island
    let parent_indices = tournament(selected_sorted_fitnesses, num_tournaments, tournament_size)

    all_selected_indices.add(selected_sorted_indices.map(idx => idx + offset))
    all_selected_fitnesses.add(selected_sorted_fitnesses)

    for i in num_selected_per_island ..< island_pop_size:
      all_target_nn_ids.add(offset + i)

    for couple in parent_indices:
      let ((p1_idx, p1_fit), (p2_idx, p2_fit)) = couple
      let global_p1_idx = selected_sorted_indices[p1_idx] + offset
      var global_p2_idx = selected_sorted_indices[p2_idx] + offset
      var final_p2_fit = p2_fit

      if hyper_params.use_queen_bee:
        global_p2_idx = selected_sorted_indices[0] + offset
        final_p2_fit = selected_sorted_fitnesses[0]

      all_parent_indices.add(((global_p1_idx, p1_fit), (global_p2_idx, final_p2_fit)))

    for top_id in top_ids:
      let top = topologies[top_id]
      let selected = selected_sorted_indices.map(index => top.population[index + offset])

      for i in 0 ..< num_selected_per_island:
        top.population[offset + i] = selected[i]

      top.curr_pop_size = num_selected_per_island * num_islands

  return (all_selected_indices, all_selected_fitnesses, all_parent_indices, all_target_nn_ids)

proc mutate(
  top: Topology,
  nn_id: int,
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.gcsafe exportpy.} =
  let hparams = mutation_hyper_params.get(top.mutation_hyper_params)

  let nn = top.population[nn_id]

  if not satisfy_prob(hparams.mutate_prob):
    return

  var node_index = init_table[int, int]()
  var edge_index = init_table[int, int]()

  # snapshot for safe iteration; use top.conn_index for existence checks
  var snapshot_conn_index = top.conn_index

  # indexing global to local

  for local_node_id, meta_node in nn.meta_nodes:
    node_index[meta_node.node_id] = local_node_id

  for local_edge_id, meta_edge in nn.meta_edges:
    edge_index[meta_edge.edge_id] = local_edge_id

  let meta_nodes_len = nn.meta_nodes.len
  let meta_edges_len = nn.meta_edges.len

  template perturb_value(val: var float32, replace_prob: float, perturb_strength: float, max_mag: float) =
    if satisfy_prob(replace_prob):
      val = random_normal()
    else:
      val += random_normal() * perturb_strength
    val = max(min(val, max_mag), -max_mag)

  template fast_ga_loop(length: int, beta: float, idx: untyped, body: untyped) =
    if length > 0:
      let rate = sample_power_law(max(length div 2, 1), beta).float / length.float
      var idx = sample_waiting_time(rate) - 1
      while idx < length:
        body
        idx += sample_waiting_time(rate)

  # mutating nodes

  for i in 0 ..< meta_nodes_len:
    let node = nn.meta_nodes[i]
    if node.disabled: continue

    if node.can_disable and satisfy_prob(hparams.add_remove_node_prob):
      node.disabled = not node.disabled

    if node.can_change_activation and satisfy_prob(hparams.change_activation_prob):
      node.activation = rand_activation()
    
    if not hparams.use_fast_ga and satisfy_prob(hparams.change_node_bias_prob):
      perturb_value(node.bias, hparams.replace_node_bias_prob, hparams.perturb_bias_strength, hparams.max_weight_magnitude)

  if hparams.use_fast_ga:
    fast_ga_loop(meta_nodes_len, hparams.fast_ga_beta, i):
      let node = nn.meta_nodes[i]
      if not node.disabled:
        perturb_value(node.bias, hparams.replace_node_bias_prob, hparams.perturb_bias_strength, hparams.max_weight_magnitude)

  # mutating edges

  for i in 0 ..< meta_edges_len:
    let edge = nn.meta_edges[i]
    if edge.disabled: continue

    if not hparams.use_fast_ga and satisfy_prob(hparams.change_edge_weight_prob):
      perturb_value(edge.weight, hparams.replace_edge_weight_prob, hparams.perturb_weight_strength, hparams.max_weight_magnitude)

    if edge.can_disable and satisfy_prob(hparams.toggle_meta_edge_prob):
      edge.disabled = not edge.disabled
      if not edge.disabled: edge.weight = random_normal()

  if hparams.use_fast_ga:
    fast_ga_loop(meta_edges_len, hparams.fast_ga_beta, i):
      let edge = nn.meta_edges[i]
      if not edge.disabled:
        perturb_value(edge.weight, hparams.replace_edge_weight_prob, hparams.perturb_weight_strength, hparams.max_weight_magnitude)

  # structural mutations - fired ONCE per individual, not per node/edge

  if satisfy_prob(hparams.grow_node_prob):
    var active_nodes: seq[int] = @[]
    for i in 0 ..< meta_nodes_len:
      if not nn.meta_nodes[i].disabled: active_nodes.add(i)
    if active_nodes.len > 0:
      let meta_node = nn.meta_nodes[sample(active_nodes)]
      let meta_node_global_id = meta_node.node_id
      let new_node_id = add_node(top)
      let new_meta_node = MetaNode(
        topology_id: top.id, node_id: new_node_id, can_disable: true, disabled: false,
        activation: if coin_flip(): relu else: sigmoid, can_change_activation: coin_flip(),
      )
      let new_local_node_id = nn.meta_nodes.len
      node_index[new_node_id] = new_local_node_id
      nn.meta_nodes.add(new_meta_node)

      var new_edge_ids: seq[int] = @[]
      for node_from_to_id, edge_id in snapshot_conn_index.pairs:
        let (from_node_id, to_node_id) = node_from_to_id
        if not (node_index.has_key(from_node_id) and node_index.has_key(to_node_id) and edge_index.has_key(edge_id)): continue
        if nn.meta_edges[edge_index[edge_id]].disabled: continue
        if to_node_id == meta_node_global_id: new_edge_ids.add(add_edge(top, from_node_id, new_node_id))
        elif from_node_id == meta_node_global_id: new_edge_ids.add(add_edge(top, new_node_id, to_node_id))

      for new_edge_id in new_edge_ids:
        let edge = top.edges[new_edge_id]
        nn.meta_edges.add(MetaEdge(
          topology_id: top.id, edge_id: new_edge_id, local_from_node_id: node_index[edge.from_node_id],
          local_to_node_id: node_index[edge.to_node_id], weight: random_normal()
        ))

  if satisfy_prob(hparams.grow_edge_prob):
    var active_edges: seq[int] = @[]
    for i in 0 ..< meta_edges_len:
      if not nn.meta_edges[i].disabled: active_edges.add(i)
    if active_edges.len > 0:
      let meta_edge = nn.meta_edges[sample(active_edges)]
      meta_edge.disabled = true
      let edge = top.edges[meta_edge.edge_id]

      var node_id: int
      var edge_id1: int
      var edge_id2: int
      if top.edge_splits.has_key(meta_edge.edge_id):
        let split_info = top.edge_splits[meta_edge.edge_id]
        node_id = split_info[0]; edge_id1 = split_info[1]; edge_id2 = split_info[2]
      else:
        node_id = add_node(top)
        edge_id1 = add_edge(top, edge.from_node_id, node_id)
        edge_id2 = add_edge(top, node_id, edge.to_node_id)
        top.edge_splits[meta_edge.edge_id] = (node_id, edge_id1, edge_id2)

      let new_local_node_id = nn.meta_nodes.len
      nn.meta_nodes.add(MetaNode(
        topology_id: top.id, node_id: node_id,
        activation: if coin_flip(): relu else: sigmoid, can_change_activation: coin_flip(),
      ))
      node_index[node_id] = new_local_node_id

      nn.meta_edges.add(MetaEdge(
        topology_id: top.id, edge_id: edge_id1, local_from_node_id: meta_edge.local_from_node_id,
        local_to_node_id: new_local_node_id, weight: 1.0
      ))
      nn.meta_edges.add(MetaEdge(
        topology_id: top.id, edge_id: edge_id2, local_from_node_id: new_local_node_id,
        local_to_node_id: meta_edge.local_to_node_id, weight: meta_edge.weight
      ))

  if satisfy_prob(hparams.add_novel_edge_prob):
    var active_nodes: seq[int] = @[]
    for i in 0 ..< nn.meta_nodes.len:
      if not nn.meta_nodes[i].disabled: active_nodes.add(i)
      
    if active_nodes.len > 1:
      let from_local = sample(active_nodes)
      var to_local = sample(active_nodes)
      var attempts = 0
      while from_local == to_local and attempts < 10:
        to_local = sample(active_nodes)
        attempts += 1
        
      if from_local != to_local:
        let from_global = nn.meta_nodes[from_local].node_id
        let to_global = nn.meta_nodes[to_local].node_id
        let rct = (from_global, to_global)
        let edge_id = if not top.conn_index.has_key(rct): add_edge(top, from_global, to_global) else: top.conn_index[rct]

        if not edge_index.has_key(edge_id):
          nn.meta_edges.add(MetaEdge(
            topology_id: top.id, edge_id: edge_id, local_from_node_id: from_local,
            local_to_node_id: to_local, weight: random_normal()
          ))
          edge_index[edge_id] = nn.meta_edges.len - 1
        else:
          nn.meta_edges[edge_index[edge_id]].disabled = false

  nn.cached_exec_trace = none(ExecTrace)

proc mutate(
  top_id: int,
  nn_id: int,
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  let top = topologies[top_id]
  mutate(top, nn_id, mutation_hyper_params)

proc mutate_all(
  all_top_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =

  for top_id in all_top_ids:
    let top = topologies[top_id]

    let hyper_params = mutation_hyper_params.get(top.mutation_hyper_params)

    assert hyper_params.num_preserve_elites < top.population.len
    for nn_id in hyper_params.num_preserve_elites ..< top.population.len:

      mutate(top, nn_id, mutation_hyper_params)

    set_population_exec_trace(top_id)

proc crossover(
  top: Topology,
  first_parent_nn_id: int,
  second_parent_nn_id: int,
  first_parent_fitness: float32,
  second_parent_fitness: float32,
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
): NeuralNetwork {.exportpy.} =

  let hyper_params = crossover_hyper_params.get(top.crossover_hyper_params)

  # parents

  let parent1 = top.population[first_parent_nn_id]
  let parent2 = top.population[second_parent_nn_id]

  # child

  var child_nodes: seq[MetaNode] = @[]
  var child_edges: seq[MetaEdge] = @[]

  # index the nodes and edges of parents 1 and 2

  proc index_meta_nodes_by_global_id(
    meta_nodes: seq[MetaNode]
  ): Table[int, MetaNode] =
    result = init_table[int, MetaNode]()

    for meta_node in meta_nodes:
      result[meta_node.node_id] = meta_node

  proc index_meta_edges_by_global_id(
    meta_edges: seq[MetaEdge]
  ): Table[int, MetaEdge] =
    result = init_table[int, MetaEdge]()

    for meta_edge in meta_edges:
      result[meta_edge.edge_id] = meta_edge

  let parent1_nodes_index = index_meta_nodes_by_global_id(parent1.meta_nodes)
  let parent2_nodes_index = index_meta_nodes_by_global_id(parent2.meta_nodes)

  let parent1_edges_index = index_meta_edges_by_global_id(parent1.meta_edges)
  let parent2_edges_index = index_meta_edges_by_global_id(parent2.meta_edges)

  let parent1_node_set = parent1_nodes_index.keys.to_seq.to_hash_set
  let parent2_node_set = parent2_nodes_index.keys.to_seq.to_hash_set

  let parent1_edge_set = parent1_edges_index.keys.to_seq.to_hash_set
  let parent2_edge_set = parent2_edges_index.keys.to_seq.to_hash_set

  # handle joint

  var joint_node_ids = (parent1_node_set * parent2_node_set).to_seq
  var joint_edge_ids = (parent1_edge_set * parent2_edge_set).to_seq

  joint_node_ids.sort()
  joint_edge_ids.sort()

  # handle disjoint
  # one of the important details - the child inherits all the disjoint / excess genes from the fitter parent. seems like an advantage for in-silico evo

  var
    disjoint_nodes_index: Table[int, MetaNode]
    disjoint_edges_index: Table[int, MetaEdge]
    disjoint_node_ids: seq[int] = @[]
    disjoint_edge_ids: seq[int] = @[]

  if satisfy_prob(hyper_params.prob_inherit_all_excess_genes):

    # add a little noise for randomly tie-breaking parent one and two when scores are identical

    let noised_first_parent_fitness = first_parent_fitness + random_normal() * 1e-2

    if noised_first_parent_fitness <= second_parent_fitness:
      disjoint_nodes_index = parent2_nodes_index
      disjoint_edges_index = parent2_edges_index
      disjoint_node_ids = (parent2_node_set - parent1_node_set).to_seq
      disjoint_edge_ids = (parent2_edge_set - parent1_edge_set).to_seq

    else:
      disjoint_nodes_index = parent1_nodes_index
      disjoint_edges_index = parent1_edges_index
      disjoint_node_ids = (parent1_node_set - parent2_node_set).to_seq
      disjoint_edge_ids = (parent1_edge_set - parent2_edge_set).to_seq

  # new child node index

  var child_node_index = init_table[int, int]()

  # handle joint nodes

  for node_id in joint_node_ids:

    let
      parent1_node = parent1_nodes_index[node_id]
      parent2_node = parent2_nodes_index[node_id]

    let rand_node = if coin_flip():
      parent1_node
    else:
      parent2_node

    let new_node = MetaNode()
    new_node[] = rand_node[]

    if (parent1_node.disabled or parent2_node.disabled):
      new_node.disabled = satisfy_prob(hyper_params.prob_child_disabled_given_parent_cond)

    # some of the time, do not inherit disabled genes

    if satisfy_prob(hyper_params.prob_remove_disabled_node) and new_node.disabled:
      continue

    child_node_index[new_node.node_id] = child_nodes.len
    child_nodes.add(new_node)

  # handle node disjoint / excess genes

  for node_id in disjoint_node_ids:
    let disjoint_node = disjoint_nodes_index[node_id]

    let new_node = MetaNode()
    new_node[] = disjoint_node[]

    child_node_index[new_node.node_id] = child_nodes.len
    child_nodes.add(new_node)

  # handle joint edges

  for edge_id in joint_edge_ids:

    let
      parent1_edge = parent1_edges_index[edge_id]
      parent2_edge = parent2_edges_index[edge_id]

    let rand_edge = if coin_flip():
      parent1_edge
    else:
      parent2_edge

    let new_edge = MetaEdge()
    new_edge[] = rand_edge[]

    if (parent1_edge.disabled or parent2_edge.disabled):
      new_edge.disabled = satisfy_prob(hyper_params.prob_child_disabled_given_parent_cond)

    let edge = top.edges[new_edge.edge_id]

    if (
      edge.from_node_id notin child_node_index or
      edge.to_node_id notin child_node_index
    ):
      continue

    new_edge.local_from_node_id = child_node_index[edge.from_node_id]
    new_edge.local_to_node_id = child_node_index[edge.to_node_id]

    child_edges.add(new_edge)

  # handle edges disjoint / excess genes

  for edge_id in disjoint_edge_ids:
    let disjoint_edge = disjoint_edges_index[edge_id]

    let new_edge = MetaEdge()
    new_edge[] = disjoint_edge[]

    let edge = top.edges[new_edge.edge_id]

    if (
      edge.from_node_id notin child_node_index or
      edge.to_node_id notin child_node_index
    ):
      continue

    new_edge.local_from_node_id = child_node_index[edge.from_node_id]
    new_edge.local_to_node_id = child_node_index[edge.to_node_id]

    child_edges.add(new_edge)

  # add child to population

  let nn = NeuralNetwork(
    num_inputs: top.num_inputs,
    num_outputs: top.num_outputs,
    num_hiddens: top.num_hiddens,
    meta_nodes: child_nodes,
    meta_edges: child_edges
  )

  return nn

proc crossover_one_couple_and_add_to_population(
  top: Topology,
  nn_id: int,
  couple: Couple,
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
) {.exportpy.} =

  let (parent1_info, parent2_info) = couple
  let (parent1, fitness1) = parent1_info
  let (parent2, fitness2) = parent2_info
  
  let sel_params = top.selection_hyper_params

  if sel_params.use_queen_bee:
    let drone_clone = crossover(top, parent1, parent1, fitness1, fitness1, crossover_hyper_params)
    top.population[nn_id] = drone_clone

    var strong_mut = top.mutation_hyper_params
    strong_mut.mutate_prob = 1.0
    strong_mut.change_edge_weight_prob = min(1.0, strong_mut.change_edge_weight_prob + sel_params.queen_strong_mutation_rate)
    strong_mut.replace_edge_weight_prob = min(1.0, strong_mut.replace_edge_weight_prob + sel_params.queen_strong_mutation_rate)
    
    mutate(top, nn_id, strong_mut.some)

    let final_child = crossover(top, nn_id, parent2, fitness1, fitness2, crossover_hyper_params)
    top.population[nn_id] = final_child
  else:
    let child = crossover(top, parent1, parent2, fitness1, fitness2, crossover_hyper_params)
    top.population[nn_id] = child

proc crossover_one_couple_and_add_to_population(
  top_id: int,
  nn_id: int,
  couple: Couple,
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
) {.exportpy.} =
  let top = topologies[top_id]
  crossover_one_couple_and_add_to_population(top, nn_id, couple, crossover_hyper_params)

proc crossover_and_add_to_population(
  top_ids: seq[int],
  couples: seq[Couple],
  target_nn_ids: seq[int] = @[],
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
) {.exportpy.} =

  for top_id in top_ids:
    let top = topologies[top_id]

    if target_nn_ids.len > 0:
      assert target_nn_ids.len == couples.len
      for i in 0 ..< couples.len:
        let couple = couples[i]
        let nn_id = target_nn_ids[i]
        crossover_one_couple_and_add_to_population(top, nn_id, couple, crossover_hyper_params)
    else:
      for item in zip(couples, (top.curr_pop_size ..< top.pop_size).to_seq):
        let (couple, nn_id) = item
        crossover_one_couple_and_add_to_population(top, nn_id, couple, crossover_hyper_params)

    top.curr_pop_size = top.pop_size

# migration and island reset

proc clone*(nn: NeuralNetwork): NeuralNetwork =
  var cloned_nodes: seq[MetaNode] = @[]
  for node in nn.meta_nodes:
    let new_node = MetaNode()
    new_node[] = node[]
    cloned_nodes.add(new_node)

  var cloned_edges: seq[MetaEdge] = @[]
  for edge in nn.meta_edges:
    let new_edge = MetaEdge()
    new_edge[] = edge[]
    cloned_edges.add(new_edge)

  return NeuralNetwork(
    id: nn.id,
    topology_id: nn.topology_id,
    num_inputs: nn.num_inputs,
    num_outputs: nn.num_outputs,
    num_hiddens: nn.num_hiddens,
    meta_nodes: cloned_nodes,
    meta_edges: cloned_edges,
    cached_exec_trace: none(ExecTrace),
  )

proc migrate_islands*(
  all_top_ids: seq[int],
  num_migrants: int
) {.exportpy.} =
  if all_top_ids.len > 0:
    let first_top = topologies[all_top_ids[0]]
    let num_islands = first_top.num_islands
    if num_islands > 1:
      let pop_size = first_top.pop_size
      let island_pop_size = pop_size div num_islands

      if num_migrants < island_pop_size:
        for top_id in all_top_ids:
          let top = topologies[top_id]
          var new_pop = new_seq[NeuralNetwork](pop_size)

          for i in 0 ..< num_islands:
            let curr_offset = i * island_pop_size
            let prev_island = (i + num_islands - 1) mod num_islands
            let prev_offset = prev_island * island_pop_size

            for j in 0 ..< island_pop_size:
              new_pop[curr_offset + j] = top.population[curr_offset + j]

            for j in 0 ..< num_migrants:
              new_pop[curr_offset + island_pop_size - 1 - j] = top.population[prev_offset + j].clone()

          top.population = new_pop

proc reset_top_islands*(
  all_top_ids: seq[int],
  fitnesses: seq[float32],
  num_islands_to_reset: int,
  tournament_size: int
) {.exportpy.} =
  if all_top_ids.len > 0:
    let first_top = topologies[all_top_ids[0]]
    let num_islands = first_top.num_islands
    if num_islands > 1:
      let pop_size = first_top.pop_size
      let island_pop_size = pop_size div num_islands

      var island_avgs: seq[(float32, int)] = @[]
      for i in 0 ..< num_islands:
        let offset = i * island_pop_size
        let island_fits = fitnesses[offset ..< offset + island_pop_size]
        var sum: float32 = 0.0
        for f in island_fits: sum += f
        island_avgs.add((sum / island_pop_size.float32, i))

      island_avgs.sort(proc (x, y: (float32, int)): int = cmp(x[0], y[0]))

      let reset_island_ids = island_avgs[0 ..< num_islands_to_reset].map(x => x[1])
      let survivor_island_ids = island_avgs[num_islands_to_reset .. ^1].map(x => x[1])

      var global_survivor_indices: seq[int] = @[]
      for island_id in survivor_island_ids:
        let offset = island_id * island_pop_size
        for j in 0 ..< island_pop_size:
          global_survivor_indices.add(offset + j)

      for island_id in reset_island_ids:
        let offset = island_id * island_pop_size

        for nn_id in offset ..< offset + island_pop_size:
          var p1, p2: int = -1
          var f1, f2: float32 = -1e10

          for _ in 0 ..< tournament_size:
            let idx = global_survivor_indices[rand(global_survivor_indices.len - 1)]
            let fit = fitnesses[idx]
            if fit > f1:
              p2 = p1; f2 = f1
              p1 = idx; f1 = fit
            elif fit > f2:
              p2 = idx; f2 = fit

          for top_id in all_top_ids:
            let top = topologies[top_id]
            let child = crossover(top, p1, p2, f1, f2)
            top.population[nn_id] = child

# quick test

when is_main_module:

  # neat

  let hyperneat_top_id = add_topology(3, 1, @[16, 16, 16])
  init_population(hyperneat_top_id, 10)
  var top = topologies[hyperneat_top_id]

  assert top.population.len == 10

  for i in 0..<100:
    let (_, _, couples, target_nn_ids) = select_and_tournament(@[hyperneat_top_id], @[1.0'f32, 2.0, 3.0, 5.0, 4.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    crossover_and_add_to_population(@[hyperneat_top_id], couples, target_nn_ids)
    mutate_all(@[hyperneat_top_id])

  save_json_to_file(hyperneat_top_id, "./population.json")
