import nimpy

import os

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
  algorithm
]

import nimpy/[
  raw_buffers,
  py_types
]

import json

import bitvector

import arraymancer

import jsony

import malebolgia

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
    res = indices[0]
    stride = shape[0]

  for axis, idx in indices[1 .. ^1]:
    res += (stride * idx)
    stride *= shape[axis + 1]

  res

# functions

proc satisfy_prob(
  prob: Prob
): bool =

  if prob == 0.0:
    return false
  elif prob == 1.0:
    return true

  return rand(1.0) < prob

proc coin_flip(): bool =
  return satisfy_prob(0.5)

proc random_normal(
  eps: float = 1e-30
): float32 =
  # box-muller for random normal
  let
    u1 = rand(1.0)
    u2 = rand(1.0)

  return sqrt(-2 * ln(max(eps, u1))) * cos(2 * PI * u2)

# activation functions

proc sigmoid(x: float32): float32 =
  return (1.0 / (1.0 + exp(-x)))

proc relu(x: float32): float32 =
  return max(x, 0.0)

proc gauss(x: float32): float32 =
  return exp(-pow(x, 2))

proc identity(x: float32): float32 =
  return x

proc elu(x: float32): float32 =
  if x >= 0.0:
    result = x
  else:
    result = exp(x) - 1.0

proc clamp_one(x: float32): float32 =
  return max(min(x, 1.0), -1.0)

type
  Activation = enum
    identity, sigmoid, tanh, relu, clamp, elu, gauss, sin, abs

# types

type
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

  ExecTrace = tuple
    node_info: NumNodesInfo
    node_updatess: seq[NodeUpdate]

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
    tournament_frac: float = 0.25

  CrossoverHyperParams = object
    prob_child_disabled_given_parent_cond: float = 0.75
    prob_remove_disabled_node: float = 0.01
    prob_inherit_all_excess_genes: float = 1.0

  MutationHyperParams = object
    mutate_prob: float = 0.95
    add_novel_edge_prob: float = 5e-3
    toggle_meta_edge_prob: float = 0.05
    add_remove_node_prob: float = 1e-5
    change_activation_prob: float = 0.001
    change_edge_weight_prob: float = 0.5
    replace_edge_weight_prob: float = 0.1    # the percentage of time to replace the edge weight wholesale, which they did in the paper in addition to perturbing
    change_node_bias_prob: float = 0.1
    replace_node_bias_prob: float = 0.1
    grow_edge_prob: float = 5e-4             # this is the mutation introduced in the seminal NEAT paper that takes an existing edge for a CPPN and disables it, replacing it with a new node plus two new edges. the afferent edge is initialized to 1, the efferent inherits same weight as the one disabled. this is something currently neural network frameworks simply cannot do, and what interests me
    grow_node_prob: float = 1e-5             # similarly, some follow up research do a variation of the above and split an existing node into two nodes, in theory this leads to the network modularization
    perturb_weight_strength: float = 0.1
    perturb_bias_strength: float = 0.1
    num_preserve_elites: int = 0

  # 'topology' - rename to population at some point

  Topology = ref object
    id: int

    nodes: seq[Node] = @[] # nodes will be always arrange [input] [output] [hiddens]
    edges: seq[Edge] = @[]

    nodes_index: Table[int, Node]
    edges_index: Table[int, Edge]
    conn_index: Table[(int, int), int]

    num_inputs: int
    num_outputs: int
    num_hiddens: seq[int] = @[]

    nn_id: int = 0
    node_innovation_id: Atomic[int]
    edge_innovation_id: Atomic[int]

    pop_size: int = 0
    curr_pop_size: int = 0
    population: seq[NeuralNetwork] = @[]

    # default hyperparams

    selection_hyper_params: SelectionHyperParams

    crossover_hyper_params: CrossoverHyperParams

    mutation_hyper_params: MutationHyperParams

  # crossover related

  ParentAndFitness = tuple
    index: int
    fitness: float32

  Couple = tuple
    parent1: ParentAndFitness
    parent2: ParentAndFitness

  Couples = seq[Couple]

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
  selection_hyper_params: SelectionHyperParams = SelectionHyperParams()
): int {.exportpy.} =

  let topology = Topology(
    id: topology_id.fetch_add(1),
    num_inputs: num_inputs,
    num_outputs: num_outputs,
    num_hiddens: num_hiddens,
    mutation_hyper_params: mutation_hyper_params,
    crossover_hyper_params: crossover_hyper_params,
    selection_hyper_params: selection_hyper_params
  )

  topology.nodes_index = init_table[int, Node]()
  topology.edges_index = init_table[int, Edge]()
  topology.conn_index = init_table[(int, int), int]()

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

  let node = Node(id: top.node_innovation_id.fetch_add(1), topology_id: top.id)
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
      activation: sigmoid
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

  return ((top.num_inputs, top.num_outputs, num_nodes), trace)

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

  var (meta_info, trace) = trace_with_meta_info
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

proc evaluate_nn_single_with_trace_thread_fn(
  trace: ptr ExecTrace,
  buffer_input: ptr UncheckedArray[float32],
  buffer_output: ptr UncheckedArray[float32]
) {.gcsafe.} =

  let num_inputs = trace[].node_info.num_inputs
  var inputs = new_seq[float32](num_inputs)

  for i in 0 ..< num_inputs:
    inputs[i] = buffer_input[i]

  var (meta_info, trace) = trace[]
  let (_, num_outputs, num_nodes) = meta_info

  var values = new_seq[float32](num_nodes)

  values &= inputs

  for update_node in trace:
    let (to_id, act_index, bias, incoming_weights) = update_node

    values[to_id] = values[to_id] + bias

    for incoming_weight in incoming_weights:
      let (weight, from_id) = incoming_weight
      values[to_id] += weight * values[from_id]

    values[to_id] = Activation(act_index).activate(values[to_id])

  # values is [inputs] [output] [hiddens] [init inputs]

  for i in 0 ..< num_outputs:
    buffer_output[i] = values[i + num_inputs]

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

  let input_first_dim = nd_array_inputs.shape[0]
  assert input_first_dim == top.pop_size

  var master = create_master()

  # using malebolgia for multi-threading

  master.await_all:
    for nn_id in 0 ..< top.pop_size:

      let nn = top.population[nn_id]

      let input_index = parse_indices(@[nn_id, 0], nd_array_inputs.shape)
      let output_index = parse_indices(@[nn_id, 0], nd_array_outputs.shape)

      # input and output buffers for thread

      let buffer_input = cast[ptr UncheckedArray[float32]](nd_array_inputs.data[input_index].addr)
      let buffer_output = cast[ptr UncheckedArray[float32]](nd_array_outputs.data[output_index].addr)

      # spawn thread

      master.spawn evaluate_nn_single_with_trace_thread_fn(
        nn.cached_exec_trace.get.addr,
        buffer_input,
        buffer_output
      )

  nd_array_inputs.release()
  nd_array_outputs.release()

proc generate_hyper_weights(
  top_id: int,
  nn_id: int,
  shape: seq[int]
): seq[float32] {.exportpy.} =

  let top = topologies[top_id]

  assert top.num_inputs == shape.len

  let
    first_axis = shape[0]
    rest_axis = shape[1..^1]

  var coors = linspace(-1.0, 1.0, shape[0]).as_type(float32)
  coors = coors.reshape(1, first_axis)

  for dim in rest_axis:
    var next_dim_coors = linspace(0.0, 1.0, dim).as_type(float32)
    next_dim_coors = next_dim_coors.reshape(1, 1, dim).broadcast(1, coors.shape[1], dim)
    coors = coors.reshape(coors.shape[0], coors.shape[1], 1).broadcast(coors.shape[0], coors.shape[1], dim)
    coors = concat(coors, next_dim_coors, axis = 0)
    coors = coors.reshape(coors.shape[0], coors.shape[1] * coors.shape[2])

  var weights = evaluate_nn(top_id, nn_id, coors.to_seq_2d).to_tensor
  let meta_data = to_metadata(shape)

  return weights.reshape(meta_data).to_seq

proc generate_all_hyper_weights(
  top_id: int,
  shape: seq[int]
): seq[seq[float32]] {.exportpy.} =
  let top = topologies[top_id]

  for nn_id in 0 ..< top.population.len:
    result.add(generate_hyper_weights(top_id, nn_id, shape))

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

proc tournament(
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
  Couples
) {.exportpy.} =

  assert top_ids.len > 0

  assert top_ids.len > 0

  let one_top_id = top_ids[0]

  let top = topologies[one_top_id]

  let pop_size = top.population.len

  let hyper_params = selection_hyper_params.get(top.selection_hyper_params)

  let num_selected = max(2, (hyper_params.frac_natural_selected * pop_size.float).int)

  let tournament_size = max(2, (hyper_params.tournament_frac * num_selected.float).int)

  assert tournament_size <= num_selected
  assert pop_size > num_selected

  # select

  let sorted_indices = fitnesses
    .to_tensor()
    .argsort(order = SortOrder.Descending)
    .to_flat_seq()

  let selected_sorted_indices = sorted_indices[0..<num_selected]

  # get fitnesses

  let selected_sorted_fitnesses = selected_sorted_indices.map(index => fitnesses[index])

  let num_tournaments = pop_size - num_selected # replenish back to original population size, 1 child per tournament

  # tournament to get parent pairs

  let parent_indices = tournament(selected_sorted_fitnesses, num_tournaments, tournament_size)

  # remove least fit for all top ids passed in

  for top_id in top_ids:
    let top = topologies[top_id]

    let selected = selected_sorted_indices.map(index => top.population[index])

    for i in 0 ..< num_selected:
      top.population[i] = selected[i]

    top.curr_pop_size = num_selected

  return (selected_sorted_indices, selected_sorted_fitnesses, parent_indices)

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

  var global_conn_index = top.conn_index

  # indexing global to local

  for local_node_id, meta_node in nn.meta_nodes:
    node_index[meta_node.node_id] = local_node_id

  for local_edge_id, meta_edge in nn.meta_edges:
    edge_index[meta_edge.edge_id] = local_edge_id

  let meta_nodes_len = nn.meta_nodes.len
  let meta_edges_len = nn.meta_edges.len

  # mutating nodes

  for local_node_id in 0 ..< meta_nodes_len:

    let meta_node = nn.meta_nodes[local_node_id]

    if meta_node.disabled:
      continue

    # maybe grow a new module

    if satisfy_prob(hparams.grow_node_prob):
      let meta_node_global_id = meta_node.node_id

      # create the new node

      let new_node_id = add_node(top)

      let new_meta_node = MetaNode(
        topology_id: top.id,
        node_id: new_node_id,
        can_disable: true,
        disabled: false,
        activation: if coin_flip(): relu else: sigmoid,
        can_change_activation: coin_flip(),
      )

      let new_local_node_id = nn.meta_nodes.len

      node_index[new_node_id] = new_local_node_id
      nn.meta_nodes.add(new_meta_node)

      # connections, afferent and efferent

      var new_edge_ids: seq[int] = @[]

      for node_from_to_id, edge_id in global_conn_index.pairs:
        let (from_node_id, to_node_id) = node_from_to_id

        # only if this individual has the prerequisite afferent and efferent neurons
        # innovate the new edges for the module

        if not (
          node_index.has_key(from_node_id) and
          node_index.has_key(to_node_id) and
          edge_index.has_key(edge_id)
        ):
          continue

        if nn.meta_edges[edge_index[edge_id]].disabled:
          continue

        if to_node_id == meta_node_global_id:
          new_edge_ids.add(add_edge(top, from_node_id, new_node_id))
        elif from_node_id == meta_node_global_id:
          new_edge_ids.add(add_edge(top, new_node_id, to_node_id))

      # add the meta edges

      for new_edge_id in new_edge_ids:
        let edge = top.edges[new_edge_id]

        let new_meta_edge = MetaEdge(
          topology_id: top.id,
          edge_id: new_edge_id,
          local_from_node_id: node_index[edge.from_node_id],
          local_to_node_id: node_index[edge.to_node_id],
          weight: random_normal()
        )

        nn.meta_edges.add(new_meta_edge)
      
    # toggling disable flag on meta node

    if meta_node.can_disable and satisfy_prob(hparams.add_remove_node_prob):
      meta_node.disabled = meta_node.disabled xor true

    # mutating an activation on a node

    if meta_node.can_change_activation and satisfy_prob(hparams.change_activation_prob):
      meta_node.activation = rand_activation()

    # mutating bias

    if satisfy_prob(hparams.change_node_bias_prob):
      if satisfy_prob(hparams.replace_node_bias_prob):
        meta_node.bias = random_normal()
      else:
        meta_node.bias += random_normal() * hparams.perturb_bias_strength

  # mutating edges

  for meta_edge_index in 0 ..< meta_edges_len:

    let meta_edge = nn.meta_edges[meta_edge_index]

    # changing a weight

    if meta_edge.disabled:
      continue

    if satisfy_prob(hparams.change_edge_weight_prob):

      if satisfy_prob(hparams.replace_edge_weight_prob):
        meta_edge.weight = random_normal()
      else:
        meta_edge.weight += random_normal() * hparams.perturb_weight_strength

    # maybe splitting an edge
    # this is the novel mutation introduced in the original NEAT paper

    if satisfy_prob(hparams.grow_edge_prob):

      # disable the edge

      meta_edge.disabled = true

      # add a new innovated node

      let node_id = add_node(top)

      # add the two innovated edges, with the new node above in between

      let edge = top.edges[meta_edge.edge_id]

      let edge_id1 = add_edge(top, edge.from_node_id, node_id)
      let edge_id2 = add_edge(top, node_id, edge.to_node_id)

      # now add the meta nodes and edges for this particular neural network instantiation

      let meta_node = MetaNode(
        topology_id: top.id,
        node_id: node_id,
        activation: if coin_flip(): relu else: sigmoid,
        can_change_activation: coin_flip(),
      )

      let new_local_node_id = nn.meta_nodes.len
      nn.meta_nodes.add(meta_node)

      let meta_edge_incoming = MetaEdge(
        topology_id: top.id,
        edge_id: edge_id1,
        local_from_node_id: meta_edge.local_from_node_id,
        local_to_node_id: new_local_node_id,
        weight: 1.0 # they initialize to 1.
      )

      nn.meta_edges.add(meta_edge_incoming)

      let meta_edge_outgoing = MetaEdge(
        topology_id: top.id,
        edge_id: edge_id2,
        local_from_node_id: new_local_node_id,
        local_to_node_id: meta_edge.local_to_node_id,
        weight: meta_edge.weight # inherits old weight
      )

      nn.meta_edges.add(meta_edge_outgoing)

    # toggling an existing edge gene in the individual

    if not (satisfy_prob(hparams.toggle_meta_edge_prob) and meta_edge.can_disable):
      continue

    meta_edge.disabled = meta_edge.disabled xor true
    if not meta_edge.disabled:
      meta_edge.weight = random_normal()

  # adding of a novel edge to the gene pool + the individual

  if satisfy_prob(hparams.add_novel_edge_prob):
    let existing_node_ids = node_index.keys
      .to_seq
      .filter(node_id => not nn.meta_nodes[node_index[node_id]].disabled)

    let cartesian_prod = product(@[existing_node_ids, existing_node_ids]).filter(pair => pair[0] != pair[1])

    let random_conn = sample(cartesian_prod)
    let (from_node_id, to_node_id) = (random_conn[0], random_conn[1])

    let random_conn_tuple = (from_node_id, to_node_id)

    let exists_in_gene_pool = global_conn_index.has_key(random_conn_tuple)

    # register novel edge in gene pool if not exists

    let edge_id = if not exists_in_gene_pool:
      add_edge(top, from_node_id, to_node_id)
    else:
      global_conn_index[random_conn_tuple]

    # # now add it to the individual

    let meta_edge = if not edge_index.has_key(edge_id):
      let new_meta_edge = MetaEdge(
        topology_id: top.id,
        edge_id: edge_id,
        local_from_node_id: node_index[from_node_id],
        local_to_node_id: node_index[to_node_id],
        weight: random_normal()
      )

      nn.meta_edges.add(new_meta_edge)
      edge_index[edge_id] = nn.meta_edges.len
      new_meta_edge
    else:
      nn.meta_edges[edge_index[edge_id]]

    meta_edge.disabled = false

  # just remove cached trace for now
  # properly detect change in future

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
  couples: seq[((int, float32), (int, float32))],
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
) {.exportpy.} =

  for top_id in top_ids:
    let top = topologies[top_id]

    for item in zip(couples, (top.curr_pop_size ..< top.pop_size).to_seq):
      let (couple, nn_id) = item
      crossover_one_couple_and_add_to_population(top, nn_id, couple, crossover_hyper_params)

    top.curr_pop_size = top.pop_size

# quick test

when is_main_module:

  # neat

  let hyperneat_top_id = add_topology(3, 1, @[16, 16, 16])
  init_population(hyperneat_top_id, 10)
  var top = topologies[hyperneat_top_id]

  assert top.population.len == 10

  for i in 0..<100:
    let (_, _, couples) = select_and_tournament(@[hyperneat_top_id], @[1.0'f32, 2.0, 3.0, 5.0, 4.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    crossover_and_add_to_population(@[hyperneat_top_id], couples)
    mutate_all(@[hyperneat_top_id])

  save_json_to_file(hyperneat_top_id, "./population.json")