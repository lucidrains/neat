import std/[
  random,
  times,
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

var
  spare_normal_ready {.threadvar.}: bool
  spare_normal {.threadvar.}: float32

proc set_seed*(seed: int64) {.exportpy.} =
  spare_normal_ready = false
  randomize(seed)

# from @Vindaar - https://github.com/yglukhov/nimpy/issues/114#issuecomment-531504502

type
  NumpyArray[T] = object
    py_buf: RawPyBuffer
    has_buf: bool
    data: ptr UncheckedArray[T]
    ndim: int
    dim0: int
    dim1: int
    len: int

proc init_nd_array[T](ar: PyObject, writable: bool = true): NumpyArray[T] {.inline.} =
  ar.get_buffer(result.py_buf, if writable: (PyBUF_WRITABLE or PyBUF_ND) else: PyBUF_ND)
  result.has_buf = true
  result.ndim = result.py_buf.ndim.int

  let shape = cast[ptr UncheckedArray[Py_ssize_t]](result.py_buf.shape)

  var total_len = 1
  for i in 0 ..< result.ndim:
    let dim = shape[i].int
    if i == 0: result.dim0 = dim
    elif i == 1: result.dim1 = dim
    total_len *= dim

  result.len = total_len
  result.data = cast[ptr UncheckedArray[T]](result.py_buf.buf)

proc release[T](nd: var NumpyArray[T]) {.inline.} =
  if nd.has_buf:
    nd.py_buf.release()
    nd.has_buf = false

# functions

template raw_ptr[T](s: openArray[T]): ptr UncheckedArray[T] =
  if s.len > 0: cast[ptr UncheckedArray[T]](s[0].unsafe_addr) else: nil

template ensure_len[T](s: var seq[T], req_len: int) =
  if s.len < req_len:
    s.set_len(req_len)

proc satisfy_prob(prob: Prob): bool {.inline.} =
  if prob == 0.0: return false
  if prob == 1.0: return true
  rand(1.0) < prob

proc coin_flip(): bool {.inline.} = rand(1.0) < 0.5

proc random_normal(): float32 {.inline.} =
  # marsaglia polar method, caching the spare sample
  if spare_normal_ready:
    spare_normal_ready = false
    return spare_normal

  var u, v, s: float
  while true:
    u = rand(2.0) - 1.0
    v = rand(2.0) - 1.0
    s = u * u + v * v
    if s > 0.0 and s < 1.0:
      break

  let scale = sqrt(-2.0 * ln(s) / s)
  spare_normal = float32(v * scale)
  spare_normal_ready = true
  return float32(u * scale)

proc clamp_magnitude(x, max_magnitude: float32): float32 {.inline.} =
  max(min(x, max_magnitude), -max_magnitude)

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

proc sample_waiting_time*(prob: float): int =
  ## geometric skip - number of trials until first success
  if prob <= 0.0: return int.high
  if prob >= 1.0: return 1
  1 + (ln(max(1.0 - rand(1.0), 1e-15)) / ln(1.0 - prob)).floor.int

# activation functions

proc sigmoid(x: float32): float32 =
  let xc = max(min(x, 60.0'f32), -60.0'f32)
  1.0'f32 / (1.0'f32 + exp(-xc))

proc relu(x: float32): float32 = max(x, 0.0'f32)
proc gauss(x: float32): float32 = exp(-min(x * x, 60.0'f32))
proc elu(x: float32): float32 = (if x >= 0.0'f32: x else: exp(max(x, -60.0'f32)) - 1.0'f32)
proc clamp_one(x: float32): float32 = max(min(x, 1.0'f32), -1.0'f32)

type
  Activation* = enum
    identity, sigmoid, tanh, relu, clamp, elu, gauss, sin, abs, cos

  # Sigma-Pi / product units & generalized propagation rules - Rumelhart, Hinton, McClelland (1986)
  Aggregation* = enum
    sum, product, max, min, mean

  NumNodesInfo = tuple
    num_inputs: int
    num_outputs: int
    num_nodes: int

  FlatEdge* = object
    weight*: float32
    from_node_id*: int
    meta_edge_index*: int

  FlatNodeUpdate* = object
    to_node_id*: int
    activation*: Activation
    aggregation*: Aggregation
    bias*: float32
    edge_start*: int
    edge_len*: int

  ExecTrace* = ref object
    node_info*: NumNodesInfo
    flat_nodes*: seq[FlatNodeUpdate]
    flat_edges*: seq[FlatEdge]
    input_activations*: seq[Activation]
    all_inputs_identity*: bool

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

  MetaNode* = ref object
    topology_id*: int
    node_id*: int
    disabled*: bool
    can_disable*: bool
    can_change_activation*: bool = true
    activation*: Activation = tanh
    can_change_aggregation*: bool = true
    aggregation*: Aggregation = sum
    bias*: float32 = 0.0
    frozen*: bool = false

  MetaEdge* = object
    topology_id*: int
    edge_id*: int
    can_disable*: bool = true
    disabled*: bool
    weight*: float32
    local_from_node_id*: int
    local_to_node_id*: int
    frozen*: bool = false

  NeuralNetwork* = ref object
    id*: int
    topology_id*: int
    num_inputs*: int
    num_outputs*: int
    num_hiddens*: seq[int] = @[]
    meta_nodes*: seq[MetaNode] = @[]
    meta_edges*: seq[MetaEdge] = @[]
    cached_exec_trace*: Option[ExecTrace]
    mutation_step_size*: float32 = 0.1

  # hyperparams

  SelectionHyperParams = object
    use_fuss: bool = false
    fuss_eps: float = 1e-5
    frac_natural_selected: float = 0.5
    tournament_size: int = 3

  CrossoverHyperParams = object
    prob_child_disabled_given_parent_cond: float = 0.75
    prob_remove_disabled_node: float = 0.01
    prob_inherit_all_excess_genes: float = 1.0

  MutationHyperParams* = object
    use_self_adaptive_mutation*: bool = false
    min_mutation_step_size*: float = 0.001
    max_mutation_step_size*: float = 0.1
    mutate_prob*: float = 0.95
    use_fast_ga*: bool = false
    fast_ga_beta*: float = 1.5
    add_novel_edge_prob*: float = 5e-3
    toggle_meta_edge_prob*: float = 0.05
    add_remove_node_prob*: float = 1e-5
    change_activation_prob*: float = 0.001
    change_aggregation_prob*: float = 0.001
    change_edge_weight_prob*: float = 0.5
    replace_edge_weight_prob*: float = 0.1
    change_node_bias_prob*: float = 0.1
    replace_node_bias_prob*: float = 0.1
    grow_edge_prob*: float = 5e-4
    grow_node_prob*: float = 1e-5
    perturb_weight_strength*: float = 0.1
    perturb_bias_strength*: float = 0.1
    num_preserve_elites*: int = 0
    max_weight_magnitude*: float = 30.0

  # 'topology' - rename to population at some point

  Topology* = ref object
    id*: int

    nodes*: seq[Node] = @[] # nodes will be always arrange [input] [output] [hiddens]
    edges*: seq[Edge] = @[]

    nodes_index*: Table[int, Node]
    edges_index*: Table[int, Edge]
    conn_index*: Table[(int, int), int]
    edge_splits*: Table[int, (int, int, int)]

    num_inputs*: int
    num_outputs*: int
    num_hiddens*: seq[int] = @[]
    num_initial_edges*: int

    nn_id*: int = 0
    node_innovation_id*: Atomic[int]
    edge_innovation_id*: Atomic[int]

    num_islands*: int = 1
    pop_size*: int = 0
    curr_pop_size*: int = 0
    population*: seq[NeuralNetwork] = @[]

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

  TopologyInfo* = object
    total_innovated_nodes*: int
    total_innovated_edges*: int

# globals

const
  # node visit states while tracing a network
  node_unvisited = 0'u8
  node_visiting = 1'u8
  node_done = 2'u8
  node_disabled = 3'u8

var topologies* = init_table[int, Topology]()
var topology_id: Atomic[int]

# helper accessors

proc get_topology_info*(
  top_id: int
): TopologyInfo {.exportpy.} =

  let top = topologies[top_id]

  result.total_innovated_nodes = top.node_innovation_id.load
  result.total_innovated_edges = top.edge_innovation_id.load

proc get_population_complexities*(
  top_id: int
): seq[float32] {.exportpy.} =

  let top = topologies[top_id]

  result = new_seq[float32]()

  for nn in top.population:
    var complexity = 0
    for node in nn.meta_nodes:
      if not node.disabled: complexity += 1
    for edge in nn.meta_edges:
      if not edge.disabled: complexity += 1
    result.add(complexity.float32)

# Clune, Mouret, Lipson (2013) connection cost & modularity
# Optimal node placement (Chklovskii 2004) minimizes sum of squared connection lengths

type
  NodeLayout = object
    x: seq[float32]
    y: seq[float32]
    degree: seq[int]
    start: seq[int]
    cursor: seq[int]
    adjacency: seq[int]
    edges: seq[(int, int)]

# thread-local scratch, reused across calls to avoid allocations

var scratch_layout {.threadvar.}: NodeLayout

proc spread_coord(i, n: int): float32 {.inline.} =
  result = if n > 1: -1.0'f32 + 2.0'f32 * float32(i) / float32(n - 1) else: 0.0'f32

proc layout_nodes(
  top: Topology,
  nn: NeuralNetwork,
  num_iters: int = 20
): int =
  {.push checks: off.}
  let num_nodes = nn.meta_nodes.len
  if num_nodes == 0: return 0

  let num_inputs = top.num_inputs
  let num_outputs = top.num_outputs
  let hidden_start = num_inputs + num_outputs

  scratch_layout.x.ensure_len(num_nodes)
  scratch_layout.y.ensure_len(num_nodes)
  scratch_layout.degree.ensure_len(num_nodes)
  scratch_layout.start.ensure_len(num_nodes + 1)
  scratch_layout.cursor.ensure_len(num_nodes)

  let x = scratch_layout.x.raw_ptr
  let y = scratch_layout.y.raw_ptr
  let degree = scratch_layout.degree.raw_ptr
  let start = scratch_layout.start.raw_ptr
  let cursor = scratch_layout.cursor.raw_ptr

  # inputs on the bottom row, outputs on the top row, hiddens in between

  for i in 0 ..< num_inputs:
    x[i] = spread_coord(i, num_inputs)
    y[i] = 0.0'f32

  for o in 0 ..< num_outputs:
    let i = num_inputs + o
    if i < num_nodes:
      x[i] = spread_coord(o, num_outputs)
      y[i] = 1.0'f32

  for i in hidden_start ..< num_nodes:
    x[i] = 0.0'f32
    y[i] = 0.5'f32

  zero_mem(degree, num_nodes * sizeof(int))
  scratch_layout.edges.ensure_len(nn.meta_edges.len)

  let edges = scratch_layout.edges.raw_ptr

  var num_edges = 0
  for edge in nn.meta_edges:
    if edge.disabled: continue

    let from_id = edge.local_from_node_id
    let to_id = edge.local_to_node_id
    if from_id < 0 or from_id >= num_nodes or to_id < 0 or to_id >= num_nodes: continue

    inc degree[from_id]
    inc degree[to_id]
    edges[num_edges] = (from_id, to_id)
    inc num_edges

  if hidden_start < num_nodes and num_edges > 0:
    var offset = 0
    for i in hidden_start ..< num_nodes:
      start[i] = offset
      cursor[i] = offset
      offset += degree[i]
    start[num_nodes] = offset

    if offset > 0:
      scratch_layout.adjacency.ensure_len(offset)
      let adjacency = scratch_layout.adjacency.raw_ptr

      # adjacency lists for hiddens only, inputs and outputs stay fixed
      for i in 0 ..< num_edges:
        let (from_id, to_id) = edges[i]
        if from_id >= hidden_start:
          adjacency[cursor[from_id]] = to_id
          inc cursor[from_id]
        if to_id >= hidden_start:
          adjacency[cursor[to_id]] = from_id
          inc cursor[to_id]

      # laplace relaxation towards the barycenter of each hidden's neighbors
      for _ in 0 ..< num_iters:
        for i in hidden_start ..< num_nodes:
          let first = start[i]
          let last = start[i + 1]
          if first == last: continue

          var sum_x, sum_y: float32
          for k in first ..< last:
            let neighbor = adjacency[k]
            sum_x += x[neighbor]
            sum_y += y[neighbor]

          let scale = 1.0'f32 / float32(last - first)
          x[i] = sum_x * scale
          y[i] = sum_y * scale

  return num_edges
  {.pop.}

proc get_population_connection_costs*(
  top_id: int,
  squared: bool = true
): seq[float32] {.exportpy.} =
  {.push checks: off.}
  let top = topologies[top_id]
  result = new_seq_of_cap[float32](top.population.len)

  for nn in top.population:
    let num_edges = top.layout_nodes(nn)
    let x = scratch_layout.x.raw_ptr
    let y = scratch_layout.y.raw_ptr
    let edges = scratch_layout.edges.raw_ptr

    var cost = 0.0'f32
    for i in 0 ..< num_edges:
      let (from_id, to_id) = edges[i]
      let dx = x[from_id] - x[to_id]
      let dy = y[from_id] - y[to_id]
      let dist_sq = dx * dx + dy * dy
      cost += (if squared: dist_sq else: sqrt(dist_sq))

    result.add(cost)
  {.pop.}

proc get_population_modularity_q*(
  top_id: int
): seq[float32] {.exportpy.} =
  {.push checks: off.}
  let top = topologies[top_id]
  result = new_seq_of_cap[float32](top.population.len)

  for nn in top.population:
    let num_edges = top.layout_nodes(nn)
    if num_edges == 0:
      result.add(0.0)
      continue

    let x = scratch_layout.x.raw_ptr
    let degree = scratch_layout.degree.raw_ptr
    let edges = scratch_layout.edges.raw_ptr
    let total_edges = num_edges.float32

    var left_edges, right_edges: float32
    var left_degree, right_degree: float32

    for i in 0 ..< nn.meta_nodes.len:
      if x[i] < 0.0:
        left_degree += degree[i].float32
      else:
        right_degree += degree[i].float32

    for i in 0 ..< num_edges:
      let (from_id, to_id) = edges[i]
      if x[from_id] < 0.0 and x[to_id] < 0.0:
        left_edges += 1.0'f32
      elif x[from_id] >= 0.0 and x[to_id] >= 0.0:
        right_edges += 1.0'f32

    let norm = 2.0'f32 * total_edges
    let left_modularity = left_edges / total_edges - (left_degree / norm) * (left_degree / norm)
    let right_modularity = right_edges / total_edges - (right_degree / norm) * (right_degree / norm)
    result.add(left_modularity + right_modularity)
  {.pop.}

# saving population to pretty json for introspecting on evolved graphs

proc skip_hook*(T: typedesc[NeuralNetwork], key: static string): bool =
  key in ["cached_exec_trace"]

proc skip_hook*(T: typedesc[Topology], key: static string): bool =
  key in ["conn_index", "edges_index", "nodes_index", "node_innovation_id", "edge_innovation_id"]

proc get_population_json*(
  top_id: int
): string {.exportpy.} =
  let top = topologies[top_id]
  return $(top.nodes, top.edges, top.population).to_json()

proc save_json_to_file*(
  top_id: int,
  filepath: string
) {.exportpy.} =

  let contents = get_population_json(top_id)

  let (dir, _, _) = split_file(filepath)
  if dir.len > 0 and not dir_exists(dir):
    create_dir(dir)

  write_file(filepath, contents.parse_json().pretty())

# functions

proc add_node(top: Topology, node_type: NodeType = hidden): int
proc add_edge(top: Topology, from_node_id: int, to_node_id: int): int

proc activate(act: Activation, input: Tensor[float32]): Tensor[float32] {.gcsafe.}
proc activate(act: Activation, input: float32): float32 {.gcsafe.}
proc activate(node: MetaNode, input: Tensor[float32]): Tensor[float32] {.gcsafe.}

proc set_population_exec_trace*(top_id: int)
proc clone*(nn: NeuralNetwork): NeuralNetwork

proc rand_activation(): Activation {.inline.}
proc rand_aggregation*(): Aggregation {.inline.}

proc add_topology*(
  num_inputs: int,
  num_outputs: int,
  num_hiddens: seq[int],
  mutation_hyper_params: MutationHyperParams = MutationHyperParams(),
  crossover_hyper_params: CrossoverHyperParams = CrossoverHyperParams(),
  selection_hyper_params: SelectionHyperParams = SelectionHyperParams(),
  num_islands: int = 1,
  num_recurrent: int = 0
): int {.exportpy.} =

  let num_initial_edges = num_inputs * num_outputs

  let topology = Topology(
    id: topology_id.fetch_add(1),
    num_inputs: num_inputs + num_recurrent,
    num_outputs: num_outputs + num_recurrent,
    num_hiddens: num_hiddens,
    num_initial_edges: num_initial_edges,
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
    total_num_inputs = num_inputs + num_recurrent
    total_num_outputs = num_outputs + num_recurrent

  for _ in 0..<total_num_inputs:
    discard add_node(topology, NodeType.input)

  for _ in 0..<total_num_outputs:
    discard add_node(topology, NodeType.output)

  # create edges

  # only connect original inputs to original outputs, excluding recurrent nodes
  let
    orig_input_node_ids = (0..<num_inputs).to_seq
    orig_output_node_ids = ((num_inputs + num_recurrent)..<(num_inputs + num_recurrent + num_outputs)).to_seq

  for input_id in orig_input_node_ids:
    for output_id in orig_output_node_ids:
      discard add_edge(topology, input_id, output_id)

  # initial pool of hidden nodes and edges, all disabled for new neural networks at start

  var hidden_node_ids: seq[seq[int]] = @[]

  for num_hidden_layer in num_hiddens:
    var layer_hidden_ids: seq[int] = @[]

    for _ in 0..<num_hidden_layer:
      layer_hidden_ids.add(add_node(topology, NodeType.hidden))

    hidden_node_ids.add(layer_hidden_ids)

  if hidden_node_ids.len > 0:
    let all_ids = @[orig_input_node_ids] & hidden_node_ids & @[orig_output_node_ids]

    for layer_index in 0 ..< all_ids.len - 1:
      let from_layer_ids = all_ids[layer_index]
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
  top.conn_index[(from_node_id, to_node_id)] = edge.id

  return edge.id

# main evolutionary functions

# population functions

proc init_nn(
  top_id: int,
  nn_id: int,
  sparsity: float32 = 0.05
) =
  {.push checks: off.}
  let top = topologies[top_id]
  let total_nodes = top.nodes.len
  let total_edges = top.edges.len

  let nn = NeuralNetwork(
    num_inputs: top.num_inputs,
    num_outputs: top.num_outputs,
    num_hiddens: top.num_hiddens,
    meta_nodes: new_seq[MetaNode](total_nodes),
    meta_edges: new_seq[MetaEdge](total_edges),
    mutation_step_size: top.mutation_hyper_params.perturb_weight_strength.float32
  )

  let
    output_node_index_start = top.num_inputs
    hidden_node_index_start = top.num_inputs + top.num_outputs
    hidden_edge_index_start = top.num_initial_edges

  # initialize input nodes

  for i in 0 ..< top.num_inputs:
    let node = top.nodes[i]
    nn.meta_nodes[i] = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: false,
      can_disable: false,
      can_change_activation: false,
      activation: identity,
      can_change_aggregation: false,
      aggregation: sum
    )

  # initial output nodes

  for i in output_node_index_start ..< hidden_node_index_start:
    let node = top.nodes[i]
    nn.meta_nodes[i] = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: false,
      can_disable: false,
      can_change_activation: false,
      activation: identity,
      can_change_aggregation: false,
      aggregation: sum
    )

  # hiddens

  for i in hidden_node_index_start ..< total_nodes:
    let node = top.nodes[i]
    nn.meta_nodes[i] = MetaNode(
      topology_id: top.id,
      node_id: node.id,
      disabled: satisfy_prob(sparsity),
      can_disable: true,
      bias: random_normal(),
      can_change_activation: true,
      activation: if coin_flip(): relu else: sigmoid,
      can_change_aggregation: true,
      aggregation: sum
    )

  # create edges - start off with only fully connected from inputs to outputs

  for i in 0 ..< hidden_edge_index_start:
    let edge = top.edges[i]
    nn.meta_edges[i] = MetaEdge(
      topology_id: top.id,
      edge_id: edge.id,
      disabled: false,
      can_disable: coin_flip(),
      local_from_node_id: edge.from_node_id,
      local_to_node_id: edge.to_node_id,
      weight: random_normal()
    )

  for i in hidden_edge_index_start ..< total_edges:
    let edge = top.edges[i]
    nn.meta_edges[i] = MetaEdge(
      topology_id: top.id,
      edge_id: edge.id,
      disabled: satisfy_prob(sparsity),
      local_from_node_id: edge.from_node_id,
      local_to_node_id: edge.to_node_id,
      weight: random_normal()
    )

  # add neural net to population

  top.population[nn_id] = nn
  {.pop.}

proc init_population*(
  top_id: int,
  pop_size: int,
) {.exportpy.} =

  assert pop_size > 0
  let top = topologies[top_id]
  assert top.pop_size == 0

  top.pop_size = pop_size
  top.curr_pop_size = pop_size
  top.population = new_seq[NeuralNetwork](pop_size)

  for nn_id in 0 ..< pop_size:
    init_nn(top_id, nn_id)

  assert top.population.len == top.pop_size

  set_population_exec_trace(top_id)

# tensor aggregation helpers

proc tensor_max(a, b: Tensor[float32]): Tensor[float32] {.inline.} =
  result = newTensor[float32](a.shape)
  for i in 0 ..< a.size:
    result[i] = max(a[i], b[i])

proc tensor_min(a, b: Tensor[float32]): Tensor[float32] {.inline.} =
  result = newTensor[float32](a.shape)
  for i in 0 ..< a.size:
    result[i] = min(a[i], b[i])

proc tensor_mul(a, b: Tensor[float32]): Tensor[float32] {.inline.} =
  result = newTensor[float32](a.shape)
  for i in 0 ..< a.size:
    result[i] = a[i] * b[i]

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

    if meta_node.disabled:
      let zero_val = zeros[float32](one_input_shape)
      finished[index] = 1
      values[index] = zero_val
      return zero_val

    # start with bias

    var next_visited = visited
    next_visited[index] = 1

    var node_value: Tensor[float32]

    # find all edges

    var input_node_index_and_weight: seq[(float32, int)] = @[] # omit visited

    for meta_edge in nn.meta_edges:
      if meta_edge.disabled:
        continue

      let local_from_node_id = meta_edge.local_from_node_id
      let local_to_node_id = meta_edge.local_to_node_id

      if local_to_node_id != index:
        continue

      let from_node = nn.meta_nodes[local_from_node_id]

      if next_visited[local_from_node_id] == 1:
        continue

      if from_node.disabled:
        continue

      input_node_index_and_weight.add((meta_edge.weight, local_from_node_id))

    # aggregate inputs to the node
    if input_node_index_and_weight.len == 0:
      node_value = zeros[float32](one_input_shape) +. meta_node.bias
    else:
      case meta_node.aggregation
      of sum:
        node_value = zeros[float32](one_input_shape) +. meta_node.bias
        for entry in input_node_index_and_weight:
          let (weight, local_from_node_id) = entry
          node_value += weight * compute_node_value(local_from_node_id, next_visited)
      of product:
        var prod_tensor = ones[float32](one_input_shape)
        for entry in input_node_index_and_weight:
          let (weight, local_from_node_id) = entry
          prod_tensor = tensor_mul(prod_tensor, weight * compute_node_value(local_from_node_id, next_visited))
        node_value = prod_tensor +. meta_node.bias
      of max:
        let (first_weight, first_from_id) = input_node_index_and_weight[0]
        var max_tensor = first_weight * compute_node_value(first_from_id, next_visited)
        for idx in 1 ..< input_node_index_and_weight.len:
          let (weight, local_from_node_id) = input_node_index_and_weight[idx]
          max_tensor = tensor_max(max_tensor, weight * compute_node_value(local_from_node_id, next_visited))
        node_value = max_tensor +. meta_node.bias
      of min:
        let (first_weight, first_from_id) = input_node_index_and_weight[0]
        var min_tensor = first_weight * compute_node_value(first_from_id, next_visited)
        for idx in 1 ..< input_node_index_and_weight.len:
          let (weight, local_from_node_id) = input_node_index_and_weight[idx]
          min_tensor = tensor_min(min_tensor, weight * compute_node_value(local_from_node_id, next_visited))
        node_value = min_tensor +. meta_node.bias
      of mean:
        var sum_tensor = zeros[float32](one_input_shape)
        for entry in input_node_index_and_weight:
          let (weight, local_from_node_id) = entry
          sum_tensor += weight * compute_node_value(local_from_node_id, next_visited)
        node_value = (sum_tensor /. input_node_index_and_weight.len.float32) +. meta_node.bias

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

type
  TraceScratch = object
    in_count: seq[int]
    in_start: seq[int]
    in_cursor: seq[int]
    edges: seq[FlatEdge]
    state: seq[uint8]

var scratch_trace {.threadvar.}: TraceScratch

proc evaluate_nn_exec_trace(
  top: Topology,
  nn_id: int,
): ExecTrace {.exportpy.} =
  {.push checks: off.}
  let nn = top.population[nn_id]
  let num_nodes = nn.meta_nodes.len
  let num_inputs = top.num_inputs
  let num_outputs = top.num_outputs

  var input_activations = new_seq[Activation](num_inputs)
  var all_inputs_identity = true
  for i in 0 ..< num_inputs:
    let activation = nn.meta_nodes[i].activation
    input_activations[i] = activation
    if activation != Activation.identity:
      all_inputs_identity = false

  scratch_trace.in_count.ensure_len(num_nodes)
  scratch_trace.in_start.ensure_len(num_nodes + 1)
  scratch_trace.in_cursor.ensure_len(num_nodes)
  scratch_trace.state.ensure_len(num_nodes)

  let in_count = scratch_trace.in_count.raw_ptr
  let in_start = scratch_trace.in_start.raw_ptr
  let in_cursor = scratch_trace.in_cursor.raw_ptr
  let state = scratch_trace.state.raw_ptr

  zero_mem(in_count, num_nodes * sizeof(int))

  # disabled nodes are excluded from the trace entirely
  for i in 0 ..< num_nodes:
    state[i] = if nn.meta_nodes[i].disabled: node_disabled else: node_unvisited

  var active_edge_count = 0
  for edge in nn.meta_edges:
    if edge.disabled: continue
    if state[edge.local_from_node_id] == node_disabled or state[edge.local_to_node_id] == node_disabled: continue
    inc in_count[edge.local_to_node_id]
    inc active_edge_count

  var offset = 0
  for i in 0 ..< num_nodes:
    in_start[i] = offset
    in_cursor[i] = offset
    offset += in_count[i]
  in_start[num_nodes] = offset

  scratch_trace.edges.ensure_len(active_edge_count)
  let trace_edges = scratch_trace.edges.raw_ptr

  for i in 0 ..< nn.meta_edges.len:
    let edge = nn.meta_edges[i]
    if edge.disabled: continue
    if state[edge.local_from_node_id] == node_disabled or state[edge.local_to_node_id] == node_disabled: continue

    let pos = in_cursor[edge.local_to_node_id]
    in_cursor[edge.local_to_node_id] = pos + 1
    trace_edges[pos] = FlatEdge(
      weight: edge.weight,
      from_node_id: edge.local_from_node_id,
      meta_edge_index: i
    )

  # inputs are ready before tracing the rest of the graph
  for i in 0 ..< num_inputs:
    state[i] = node_done

  var flat_nodes = new_seq_of_cap[FlatNodeUpdate](max(num_nodes - num_inputs, 1))
  var flat_edges = new_seq_of_cap[FlatEdge](active_edge_count)

  proc visit(index: int) =
    if state[index] == node_done:
      return

    if state[index] == node_disabled:
      state[index] = node_done
      flat_nodes.add(FlatNodeUpdate(
        to_node_id: index,
        activation: identity,
        aggregation: sum,
        bias: 0.0'f32,
        edge_start: flat_edges.len,
        edge_len: 0
      ))
      return

    let meta_node = nn.meta_nodes[index]
    state[index] = node_visiting

    let first = in_start[index]
    let last = in_start[index + 1]

    for k in first ..< last:
      let from_id = trace_edges[k].from_node_id
      if state[from_id] == node_unvisited:
        visit(from_id)

    # drop edges back into the current path, they are recurrent connections
    let edge_start = flat_edges.len
    for k in first ..< last:
      let edge = trace_edges[k]
      if state[edge.from_node_id] != node_visiting:
        flat_edges.add(edge)

    state[index] = node_done
    flat_nodes.add(FlatNodeUpdate(
      to_node_id: index,
      activation: meta_node.activation,
      aggregation: meta_node.aggregation,
      bias: meta_node.bias,
      edge_start: edge_start,
      edge_len: flat_edges.len - edge_start
    ))

  for i in 0 ..< num_outputs:
    visit(num_inputs + i)

  return ExecTrace(
    node_info: (num_inputs, num_outputs, num_nodes),
    flat_nodes: flat_nodes,
    flat_edges: flat_edges,
    input_activations: input_activations,
    all_inputs_identity: all_inputs_identity
  )
  {.pop.}

proc evaluate_nn_exec_trace(
  top_id: int,
  nn_id: int,
): ExecTrace {.exportpy.} =
  let top = topologies[top_id]
  return evaluate_nn_exec_trace(top, nn_id)

proc ensure_exec_trace(
  top: Topology,
  nn_id: int
): ExecTrace {.inline.} =
  let nn = top.population[nn_id]
  if nn.cached_exec_trace.is_none:
    nn.cached_exec_trace = evaluate_nn_exec_trace(top, nn_id).some
  return nn.cached_exec_trace.get

proc sync_exec_trace_weights(nn: NeuralNetwork) {.inline.} =
  if nn.cached_exec_trace.is_none:
    return

  {.push checks: off.}
  let trace = nn.cached_exec_trace.get

  for node in trace.flat_nodes.mitems:
    if not nn.meta_nodes[node.to_node_id].disabled:
      node.bias = nn.meta_nodes[node.to_node_id].bias

  for edge in trace.flat_edges.mitems:
    if edge.meta_edge_index >= 0:
      edge.weight = nn.meta_edges[edge.meta_edge_index].weight
  {.pop.}

proc clone_exec_trace(et: ExecTrace): ExecTrace {.inline.} =
  ExecTrace(
    node_info: et.node_info,
    flat_nodes: et.flat_nodes,
    flat_edges: et.flat_edges,
    input_activations: et.input_activations,
    all_inputs_identity: et.all_inputs_identity
  )

proc evaluate_nn_with_trace(
  trace: ExecTrace,
  inputs: seq[seq[float32]]
): seq[seq[float32]] {.gcsafe exportpy.} =
  let (num_inputs, num_outputs, num_nodes) = trace.node_info
  let input_tensors = inputs.map(seq_float32 => seq_float32.to_tensor)
  let input_shape = input_tensors[0].shape

  var values = new_seq_with(num_nodes, zeros[float32](input_shape))
  for i in 0 ..< num_inputs:
    values[i] = trace.input_activations[i].activate(input_tensors[i])

  for node in trace.flat_nodes:
    let to_id = node.to_node_id
    let first = node.edge_start
    let last = first + node.edge_len
    let bias = node.bias

    if node.edge_len == 0:
      values[to_id] = zeros[float32](input_shape) +. bias
    else:
      case node.aggregation
      of sum:
        values[to_id] = zeros[float32](input_shape) +. bias
        for j in first ..< last:
          let edge = trace.flat_edges[j]
          values[to_id] += edge.weight * values[edge.from_node_id]
      of product:
        var product = ones[float32](input_shape)
        for j in first ..< last:
          let edge = trace.flat_edges[j]
          product = tensor_mul(product, edge.weight * values[edge.from_node_id])
        values[to_id] = product +. bias
      of max:
        let first_edge = trace.flat_edges[first]
        var maximum = first_edge.weight * values[first_edge.from_node_id]
        for j in first + 1 ..< last:
          let edge = trace.flat_edges[j]
          maximum = tensor_max(maximum, edge.weight * values[edge.from_node_id])
        values[to_id] = maximum +. bias
      of min:
        let first_edge = trace.flat_edges[first]
        var minimum = first_edge.weight * values[first_edge.from_node_id]
        for j in first + 1 ..< last:
          let edge = trace.flat_edges[j]
          minimum = tensor_min(minimum, edge.weight * values[edge.from_node_id])
        values[to_id] = minimum +. bias
      of mean:
        var total = zeros[float32](input_shape)
        for j in first ..< last:
          let edge = trace.flat_edges[j]
          total += edge.weight * values[edge.from_node_id]
        values[to_id] = (total /. node.edge_len.float32) +. bias

    values[to_id] = node.activation.activate(values[to_id])

  let outputs = values[num_inputs ..< num_inputs + num_outputs]
  return outputs.map(tensor => tensor.to_seq)

proc aggregate_edges(
  aggregation: Aggregation,
  bias: float32,
  edges: ptr UncheckedArray[FlatEdge],
  values: ptr UncheckedArray[float32],
  edge_start, edge_len: int
): float32 {.inline.} =
  if edge_len == 0:
    return bias

  let stop = edge_start + edge_len

  case aggregation
  of sum:
    result = bias
    for i in edge_start ..< stop:
      let edge = edges[i]
      result += edge.weight * values[edge.from_node_id]
  of product:
    result = 1.0'f32
    for i in edge_start ..< stop:
      let edge = edges[i]
      result *= edge.weight * values[edge.from_node_id]
    result += bias
  of max:
    result = edges[edge_start].weight * values[edges[edge_start].from_node_id]
    for i in edge_start + 1 ..< stop:
      let edge = edges[i]
      result = max(result, edge.weight * values[edge.from_node_id])
    result += bias
  of min:
    result = edges[edge_start].weight * values[edges[edge_start].from_node_id]
    for i in edge_start + 1 ..< stop:
      let edge = edges[i]
      result = min(result, edge.weight * values[edge.from_node_id])
    result += bias
  of mean:
    result = 0.0'f32
    for i in edge_start ..< stop:
      let edge = edges[i]
      result += edge.weight * values[edge.from_node_id]
    result = result / float32(edge_len) + bias

var scratch_forward_values {.threadvar.}: seq[float32]

proc evaluate_nn_single_with_trace_thread_fn(
  trace_ptr: pointer,
  inputs: ptr UncheckedArray[float32],
  outputs: ptr UncheckedArray[float32]
) {.gcsafe.} =
  {.push checks: off.}
  let trace = cast[ExecTrace](trace_ptr)
  let (num_inputs, num_outputs, num_nodes) = trace.node_info

  scratch_forward_values.ensure_len(num_nodes)

  let values = scratch_forward_values.raw_ptr
  zero_mem(values, num_nodes * sizeof(float32))

  if trace.all_inputs_identity:
    copy_mem(values, inputs, num_inputs * sizeof(float32))
  else:
    for i in 0 ..< num_inputs:
      values[i] = activate(trace.input_activations[i], inputs[i])

  let nodes = trace.flat_nodes.raw_ptr
  let edges = trace.flat_edges.raw_ptr

  for i in 0 ..< trace.flat_nodes.len:
    let node = nodes[i]
    let value = aggregate_edges(node.aggregation, node.bias, edges, values, node.edge_start, node.edge_len)
    values[node.to_node_id] = activate(node.activation, value)

  for i in 0 ..< num_outputs:
    outputs[i] = values[num_inputs + i]
  {.pop.}

proc evaluate_nn_single*(
  top_id: int,
  nn_id: int,
  inputs: seq[float32],
  use_exec_cache: bool = false
): seq[float32] {.exportpy.} =
  let top = topologies[top_id]
  assert inputs.len == top.num_inputs

  if not use_exec_cache:
    let outputs = evaluate_nn(top_id, nn_id, inputs.map(value => @[value]))
    return outputs.map(tensor => tensor[0])

  let trace = ensure_exec_trace(top, nn_id)
  result = new_seq[float32](top.num_outputs)
  if top.num_outputs > 0 and inputs.len > 0:
    evaluate_nn_single_with_trace_thread_fn(cast[pointer](trace), inputs.raw_ptr, result.raw_ptr)

proc evaluate_nn_single_into*(
  top_id: int,
  nn_id: int,
  inputs: PyObject,
  outputs: PyObject
) {.exportpy.} =
  let top = topologies[top_id]
  let trace = ensure_exec_trace(top, nn_id)

  var nd_inputs = init_nd_array[float32](inputs, writable = false)
  var nd_outputs = init_nd_array[float32](outputs, writable = true)
  try:
    assert nd_inputs.len == top.num_inputs
    assert nd_outputs.len == top.num_outputs
    evaluate_nn_single_with_trace_thread_fn(cast[pointer](trace), nd_inputs.data, nd_outputs.data)
  finally:
    nd_inputs.release()
    nd_outputs.release()

proc set_population_exec_trace*(
  top_id: int
) {.exportpy.} =
  let top = topologies[top_id]
  for nn_id in 0 ..< top.population.len:
    discard ensure_exec_trace(top, nn_id)

proc evaluate_population_chunk(
  population: ptr UncheckedArray[NeuralNetwork],
  first_nn: int,
  last_nn: int,
  num_inputs: int,
  num_outputs: int,
  data_in: ptr UncheckedArray[float32],
  data_out: ptr UncheckedArray[float32]
) {.gcsafe.} =
  {.push checks: off.}
  for nn_id in first_nn ..< last_nn:
    let trace = cast[pointer](population[nn_id].cached_exec_trace.get)
    let inputs = cast[ptr UncheckedArray[float32]](data_in[nn_id * num_inputs].addr)
    let outputs = cast[ptr UncheckedArray[float32]](data_out[nn_id * num_outputs].addr)
    evaluate_nn_single_with_trace_thread_fn(trace, inputs, outputs)
  {.pop.}

const
  parallel_edge_threshold = 1 shl 16
  max_parallel_chunks = 8

proc evaluate_population(
  top_id: int,
  inputs: PyObject,
  outputs: PyObject
) {.exportpy.} =
  let top = topologies[top_id]

  var nd_inputs = init_nd_array[float32](inputs, writable = false)
  var nd_outputs = init_nd_array[float32](outputs, writable = true)

  try:
    assert nd_inputs.ndim == 2
    assert nd_outputs.ndim == 2
    assert nd_inputs.dim0 == top.pop_size
    assert nd_inputs.dim1 == top.num_inputs
    assert nd_outputs.dim0 == top.pop_size
    assert nd_outputs.dim1 == top.num_outputs

    let num_inputs = top.num_inputs
    let num_outputs = top.num_outputs
    let pop_size = top.pop_size

    var total_edges = 0
    for nn_id in 0 ..< pop_size:
      total_edges += ensure_exec_trace(top, nn_id).flat_edges.len

    let population = top.population.raw_ptr

    if total_edges < parallel_edge_threshold or pop_size < 32:
      evaluate_population_chunk(
        population, 0, pop_size, num_inputs, num_outputs,
        nd_inputs.data, nd_outputs.data
      )
    else:
      let num_chunks = min(pop_size, max_parallel_chunks)
      let chunk_size = (pop_size + num_chunks - 1) div num_chunks

      master.await_all:
        for chunk in 0 ..< num_chunks:
          let first_nn = chunk * chunk_size
          let last_nn = min(first_nn + chunk_size, pop_size)
          if first_nn < last_nn:
            master.spawn evaluate_population_chunk(
              population, first_nn, last_nn, num_inputs, num_outputs,
              nd_inputs.data, nd_outputs.data
            )
  finally:
    nd_inputs.release()
    nd_outputs.release()

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

  case act
  of identity: return input
  of sigmoid: return sigmoid(input)
  of tanh: return tanh(input)
  of relu: return relu(input)
  of clamp: return clamp_one(input)
  of elu: return elu(input)
  of gauss: return gauss(input)
  of sin: return sin(input)
  of abs: return abs(input)
  of cos: return cos(input)

proc rand_activation(): Activation {.inline.} =
  Activation(rand(Activation.high.ord))

proc rand_aggregation*(): Aggregation {.inline.} =
  Aggregation(rand(Aggregation.high.ord))

proc d_activate(act: Activation, x: float32, y: float32): float32 {.inline.} =
  case act
  of identity: 1.0
  of sigmoid:  y * (1.0 - y)
  of tanh:     1.0 - y * y
  of relu:     (if x > 0.0: 1.0'f32 else: 0.0'f32)
  of clamp:    (if x >= -1.0 and x <= 1.0: 1.0'f32 else: 0.0'f32)
  of elu:      (if x >= 0.0: 1.0'f32 else: y + 1.0)
  of gauss:    -2.0 * x * y
  of sin:      cos(x)
  of cos:      -sin(x)
  of abs:      (if x > 0.0: 1.0'f32 elif x < 0.0: -1.0'f32 else: 0.0'f32)

type
  BackpropScratch = object
    values: seq[float32]
    pre_activation: seq[float32]
    gradients: seq[float32]
    prefix_products: seq[float32]
    suffix_products: seq[float32]

var scratch_backprop {.threadvar.}: BackpropScratch

proc backprop_nn_single_raw(
  top_id: int,
  nn_id: int,
  inputs: ptr UncheckedArray[float32],
  targets: ptr UncheckedArray[float32],
  learning_rate: float32 = 0.01
) =
  {.push checks: off.}
  let top = topologies[top_id]
  let nn = top.population[nn_id]
  let trace = ensure_exec_trace(top, nn_id)
  let (num_inputs, num_outputs, num_nodes) = trace.node_info

  scratch_backprop.values.ensure_len(num_nodes)
  scratch_backprop.pre_activation.ensure_len(num_nodes)
  scratch_backprop.gradients.ensure_len(num_nodes)

  let values = scratch_backprop.values.raw_ptr
  let pre_activation = scratch_backprop.pre_activation.raw_ptr
  let gradients = scratch_backprop.gradients.raw_ptr

  if num_nodes > 0:
    zero_mem(values, num_nodes * sizeof(float32))
    zero_mem(pre_activation, num_nodes * sizeof(float32))
    zero_mem(gradients, num_nodes * sizeof(float32))

  if trace.all_inputs_identity:
    copy_mem(values, inputs, num_inputs * sizeof(float32))
    copy_mem(pre_activation, inputs, num_inputs * sizeof(float32))
  else:
    for i in 0 ..< num_inputs:
      pre_activation[i] = inputs[i]
      values[i] = activate(trace.input_activations[i], inputs[i])

  # forward

  let edges = trace.flat_edges.raw_ptr

  for node in trace.flat_nodes:
    let z = aggregate_edges(node.aggregation, node.bias, edges, values, node.edge_start, node.edge_len)
    pre_activation[node.to_node_id] = z
    values[node.to_node_id] = activate(node.activation, z)

  # backward

  for i in 0 ..< num_outputs:
    gradients[num_inputs + i] = values[num_inputs + i] - targets[i]

  let max_magnitude = top.mutation_hyper_params.max_weight_magnitude.float32

  for i in countdown(trace.flat_nodes.high, 0):
    let node = trace.flat_nodes[i]
    let to_id = node.to_node_id
    if to_id < num_inputs or nn.meta_nodes[to_id].disabled:
      continue

    let g = gradients[to_id] * d_activate(node.activation, pre_activation[to_id], values[to_id])
    let edge_start = node.edge_start
    let edge_len = node.edge_len

    if not nn.meta_nodes[to_id].frozen:
      let bias = clamp_magnitude(node.bias - learning_rate * g, max_magnitude)
      trace.flat_nodes[i].bias = bias
      nn.meta_nodes[to_id].bias = bias

    template update_edge(j: int, factor = 1.0'f32) =
      let edge = trace.flat_edges[j]
      let from_id = edge.from_node_id

      gradients[from_id] += g * edge.weight * factor

      if not nn.meta_edges[edge.meta_edge_index].frozen:
        let weight = clamp_magnitude(edge.weight - learning_rate * g * values[from_id] * factor, max_magnitude)
        trace.flat_edges[j].weight = weight
        nn.meta_edges[edge.meta_edge_index].weight = weight

    case node.aggregation
    of sum:
      for j in edge_start ..< edge_start + edge_len:
        update_edge(j)
    of mean:
      let scale = 1.0'f32 / max(edge_len.float32, 1.0'f32)
      for j in edge_start ..< edge_start + edge_len:
        update_edge(j, scale)
    of max, min:
      if edge_len > 0:
        var best_index = edge_start
        var best_value = trace.flat_edges[edge_start].weight * values[trace.flat_edges[edge_start].from_node_id]

        for j in edge_start + 1 ..< edge_start + edge_len:
          let edge = trace.flat_edges[j]
          let value = edge.weight * values[edge.from_node_id]
          if (node.aggregation == max and value > best_value) or (node.aggregation == min and value < best_value):
            best_value = value
            best_index = j

        update_edge(best_index)
    of product:
      if edge_len > 0:
        scratch_backprop.prefix_products.ensure_len(edge_len)
        scratch_backprop.suffix_products.ensure_len(edge_len)

        scratch_backprop.prefix_products[0] = 1.0'f32
        for j in 1 ..< edge_len:
          let edge = trace.flat_edges[edge_start + j - 1]
          let factor = edge.weight * values[edge.from_node_id]
          scratch_backprop.prefix_products[j] = scratch_backprop.prefix_products[j - 1] * factor

        scratch_backprop.suffix_products[edge_len - 1] = 1.0'f32
        for j in countdown(edge_len - 2, 0):
          let edge = trace.flat_edges[edge_start + j + 1]
          let factor = edge.weight * values[edge.from_node_id]
          scratch_backprop.suffix_products[j] = scratch_backprop.suffix_products[j + 1] * factor

        for j in 0 ..< edge_len:
          update_edge(edge_start + j, scratch_backprop.prefix_products[j] * scratch_backprop.suffix_products[j])
  {.pop.}

proc backprop_nn_single*(
  top_id: int,
  nn_id: int,
  inputs: seq[float32],
  targets: seq[float32],
  learning_rate: float32 = 0.01
) {.exportpy.} =
  let top = topologies[top_id]
  assert inputs.len == top.num_inputs
  assert targets.len == top.num_outputs

  backprop_nn_single_raw(top_id, nn_id, inputs.raw_ptr, targets.raw_ptr, learning_rate)

proc backprop_nn_single_buf*(
  top_id: int,
  nn_id: int,
  inputs: PyObject,
  targets: PyObject,
  learning_rate: float32 = 0.01
) {.exportpy.} =
  let top = topologies[top_id]

  var nd_inputs = init_nd_array[float32](inputs, writable = false)
  var nd_targets = init_nd_array[float32](targets, writable = false)
  try:
    assert nd_inputs.len == top.num_inputs
    assert nd_targets.len == top.num_outputs
    backprop_nn_single_raw(top_id, nn_id, nd_inputs.data, nd_targets.data, learning_rate)
  finally:
    nd_inputs.release()
    nd_targets.release()

# mutation and crossover

proc fuss_couples*(fitnesses: seq[float32], num_couples: int, eps = 1e-5'f32): Couples =
  let pop_size = fitnesses.len

  var sorted = new_seq[(float32, int)](pop_size)
  for i, f in fitnesses: sorted[i] = (f, i)
  sorted.sort()

  result = new_seq_of_cap[Couple](num_couples)

  if pop_size == 1 or sorted[0][0] == sorted[^1][0]:
    for _ in 0 ..< num_couples:
      let p1 = rand(pop_size - 1)
      let p2 = rand(pop_size - 1)
      result.add(((p1, fitnesses[p1]), (p2, fitnesses[p2])))
    return

  var cdf = new_seq[float32](pop_size)
  var sum = 0.0'f32
  for i in 0 ..< pop_size:
    let left = if i == 0: sorted[0][0] else: sorted[i - 1][0]
    let right = if i == pop_size - 1: sorted[^1][0] else: sorted[i + 1][0]
    sum += (right - left) / 2.0'f32 + eps
    cdf[i] = sum

  for _ in 0 ..< num_couples:
    let p1 = sorted[min(lower_bound(cdf, rand(sum)), pop_size - 1)][1]
    let p2 = sorted[min(lower_bound(cdf, rand(sum)), pop_size - 1)][1]
    result.add(((p1, fitnesses[p1]), (p2, fitnesses[p2])))

proc tournament*(
  fitnesses: seq[float32],
  num_tournaments: int,
  tournament_size: int
): Couples {.exportpy.} =

  assert num_tournaments > 0
  assert tournament_size >= 2
  let n = fitnesses.len
  assert n >= tournament_size

  result = new_seq_of_cap[Couple](num_tournaments)
  var gene_ids = (0 ..< n).to_seq

  for _ in 0 ..< num_tournaments:
    var
      parent1, parent2: int = -1
      fitness1, fitness2: float32 = low(float32)

    # partial fisher-yates, sampling parents without replacement
    for i in 0 ..< tournament_size:
      let j = rand(i ..< n)
      swap(gene_ids[i], gene_ids[j])
      let gene_id = gene_ids[i]
      let gene_fitness = fitnesses[gene_id]

      if gene_fitness > fitness1:
        parent2 = parent1
        fitness2 = fitness1
        parent1 = gene_id
        fitness1 = gene_fitness
      elif gene_fitness > fitness2:
        parent2 = gene_id
        fitness2 = gene_fitness

    if parent1 < 0:
      parent1 = gene_ids[0]
      fitness1 = fitnesses[parent1]
    if parent2 < 0:
      parent2 = gene_ids[1]
      fitness2 = fitnesses[parent2]

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

  let num_tournaments = island_pop_size - num_selected_per_island
  var all_selected_indices = new_seq_of_cap[int](num_selected_per_island * num_islands)
  var all_selected_fitnesses = new_seq_of_cap[float32](num_selected_per_island * num_islands)
  var all_parent_indices = new_seq_of_cap[Couple](num_tournaments * num_islands)
  var all_target_nn_ids = new_seq_of_cap[int](num_tournaments * num_islands)

  var sorted_indices = new_seq[int](island_pop_size)
  var selected_sorted_fitnesses = new_seq[float32](num_selected_per_island)
  var island_pop_buf = new_seq[NeuralNetwork](island_pop_size)

  for island_id in 0 ..< num_islands:
    let offset = island_id * island_pop_size
    for i in 0 ..< island_pop_size:
      sorted_indices[i] = i

    sorted_indices.sort(proc (a, b: int): int =
      cmp(fitnesses[offset + b], fitnesses[offset + a])
    )

    for i in 0 ..< num_selected_per_island:
      let idx = sorted_indices[i]
      let fit = fitnesses[offset + idx]
      selected_sorted_fitnesses[i] = fit
      all_selected_indices.add(offset + idx)
      all_selected_fitnesses.add(fit)

    var parent_indices: Couples
    if hyper_params.use_fuss:
      parent_indices = fuss_couples(selected_sorted_fitnesses, num_tournaments, hyper_params.fuss_eps.float32)
    else:
      parent_indices = tournament(selected_sorted_fitnesses, num_tournaments, tournament_size)

    for i in num_selected_per_island ..< island_pop_size:
      all_target_nn_ids.add(offset + i)

    for couple in parent_indices:
      let ((p1_idx, p1_fit), (p2_idx, p2_fit)) = couple
      all_parent_indices.add(((offset + p1_idx, p1_fit), (offset + p2_idx, p2_fit)))

    for top_id in top_ids:
      let t = topologies[top_id]
      for i in 0 ..< island_pop_size:
        island_pop_buf[i] = t.population[offset + sorted_indices[i]]
      for i in 0 ..< island_pop_size:
        t.population[offset + i] = island_pop_buf[i]
      t.curr_pop_size = num_selected_per_island * num_islands

  return (all_selected_indices, all_selected_fitnesses, all_parent_indices, all_target_nn_ids)

template geom_loop(start_idx, end_idx: int, prob: float, idx: untyped, body: untyped) =
  if prob > 0.0 and start_idx < end_idx:
    if prob >= 0.25:
      for idx in start_idx ..< end_idx:
        if satisfy_prob(prob):
          body
    else:
      let inv_ln = 1.0 / ln(1.0 - prob)
      var idx = start_idx + int(floor(ln(max(1.0 - rand(1.0), 1e-15)) * inv_ln))
      while idx < end_idx:
        body
        idx += 1 + int(floor(ln(max(1.0 - rand(1.0), 1e-15)) * inv_ln))

proc mutate_weights*(
  top: Topology,
  nn_id: int,
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none,
  sync_trace: bool = true
) =
  {.push checks: off.}
  let hparams = mutation_hyper_params.get(top.mutation_hyper_params)

  let nn = top.population[nn_id]

  if not satisfy_prob(hparams.mutate_prob):
    return

  let meta_nodes_len = nn.meta_nodes.len
  let meta_edges_len = nn.meta_edges.len

  # determine mutation step sizes

  var
    weight_step = hparams.perturb_weight_strength.float32
    bias_step   = hparams.perturb_bias_strength.float32

  if hparams.use_self_adaptive_mutation:
    var num_active_genes = 0
    for i in 0 ..< meta_nodes_len:
      if not nn.meta_nodes[i].disabled: inc num_active_genes
    for i in 0 ..< meta_edges_len:
      if not nn.meta_edges[i].disabled: inc num_active_genes

    if num_active_genes > 0:
      let tau = 1.0'f32 / sqrt(num_active_genes.float32)
      let step_multiplier = exp(tau * random_normal())

      nn.mutation_step_size *= step_multiplier
      let
        min_step = hparams.min_mutation_step_size.float32
        max_step = hparams.max_mutation_step_size.float32
      nn.mutation_step_size = max(min(nn.mutation_step_size, max_step), min_step)

    weight_step = nn.mutation_step_size
    bias_step   = nn.mutation_step_size

  let
    replace_bias_prob = hparams.replace_node_bias_prob
    replace_edge_prob = hparams.replace_edge_weight_prob
    max_magnitude = hparams.max_weight_magnitude

  template perturb_value(value: var float32, replace_prob: float, perturb_strength: float, max_magnitude: float) =
    if satisfy_prob(replace_prob):
      value = random_normal()
    else:
      value += random_normal() * perturb_strength.float32
    value = clamp_magnitude(value, max_magnitude.float32)

  template fast_ga_loop(length: int, beta: float, idx: untyped, body: untyped) =
    if length > 0:
      let rate = sample_power_law(max(length div 2, 1), beta).float / length.float
      var idx = sample_waiting_time(rate) - 1
      while idx < length:
        body
        idx += sample_waiting_time(rate)

  # mutating node biases (only non-input nodes use biases in forward evaluation)

  if hparams.use_fast_ga:
    if meta_nodes_len > nn.num_inputs:
      fast_ga_loop(meta_nodes_len - nn.num_inputs, hparams.fast_ga_beta, rel_i):
        let node = nn.meta_nodes[nn.num_inputs + rel_i]
        if not node.disabled and not node.frozen:
          perturb_value(node.bias, replace_bias_prob, bias_step.float, max_magnitude)

    fast_ga_loop(meta_edges_len, hparams.fast_ga_beta, i):
      if not nn.meta_edges[i].disabled and not nn.meta_edges[i].frozen:
        perturb_value(nn.meta_edges[i].weight, replace_edge_prob, weight_step.float, max_magnitude)
  else:
    geom_loop(nn.num_inputs, meta_nodes_len, hparams.change_node_bias_prob, i):
      let node = nn.meta_nodes[i]
      if not node.disabled and not node.frozen:
        perturb_value(node.bias, replace_bias_prob, bias_step.float, max_magnitude)

    geom_loop(0, meta_edges_len, hparams.change_edge_weight_prob, i):
      if not nn.meta_edges[i].disabled and not nn.meta_edges[i].frozen:
        perturb_value(nn.meta_edges[i].weight, replace_edge_prob, weight_step.float, max_magnitude)

  if sync_trace:
    sync_exec_trace_weights(nn)
  {.pop.}

proc mutate_structure*(
  top: Topology,
  nn_id: int,
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none,
  force: bool = false
): bool {.discardable.} =
  {.push checks: off.}
  let hparams = mutation_hyper_params.get(top.mutation_hyper_params)

  let nn = top.population[nn_id]

  if not force and not satisfy_prob(hparams.mutate_prob):
    return false

  let meta_nodes_len = nn.meta_nodes.len
  let meta_edges_len = nn.meta_edges.len

  var mutated = false

  # mutating nodes: enable/disable and activation changes

  geom_loop(0, meta_nodes_len, hparams.add_remove_node_prob, i):
    let node = nn.meta_nodes[i]
    if not node.frozen and node.can_disable:
      node.disabled = not node.disabled
      mutated = true

  geom_loop(0, meta_nodes_len, hparams.change_activation_prob, i):
    let node = nn.meta_nodes[i]
    if not node.frozen and not node.disabled and node.can_change_activation:
      node.activation = rand_activation()
      mutated = true

  geom_loop(0, meta_nodes_len, hparams.change_aggregation_prob, i):
    let node = nn.meta_nodes[i]
    if not node.frozen and not node.disabled and node.can_change_aggregation:
      node.aggregation = rand_aggregation()
      mutated = true

  # mutating edges: enable/disable toggle

  geom_loop(0, meta_edges_len, hparams.toggle_meta_edge_prob, i):
    if not nn.meta_edges[i].frozen and nn.meta_edges[i].can_disable:
      nn.meta_edges[i].disabled = not nn.meta_edges[i].disabled
      if not nn.meta_edges[i].disabled: nn.meta_edges[i].weight = random_normal()
      mutated = true

  proc find_local_node_id(nn: NeuralNetwork, global_node_id: int): int {.inline.} =
    for i in 0 ..< nn.meta_nodes.len:
      if nn.meta_nodes[i].node_id == global_node_id:
        return i
    return -1

  proc find_local_edge_id(nn: NeuralNetwork, global_edge_id: int): int {.inline.} =
    for i in 0 ..< nn.meta_edges.len:
      if nn.meta_edges[i].edge_id == global_edge_id:
        return i
    return -1

  proc new_hidden_node(global_node_id: int): MetaNode =
    MetaNode(
      topology_id: top.id,
      node_id: global_node_id,
      can_disable: true,
      activation: if coin_flip(): relu else: sigmoid,
      aggregation: sum
    )

  proc add_meta_edge(global_edge_id, from_node, to_node: int, weight: float32) =
    nn.meta_edges.add(MetaEdge(
      topology_id: top.id,
      edge_id: global_edge_id,
      local_from_node_id: from_node,
      local_to_node_id: to_node,
      weight: weight
    ))

  template do_grow_node() =
    var active_nodes: seq[int] = @[]
    for i in 0 ..< meta_nodes_len:
      let node = nn.meta_nodes[i]
      if node.disabled: continue
      if top.nodes[node.node_id].`type` != NodeType.hidden: continue
      active_nodes.add(i)

    if active_nodes.len > 0:
      let chosen_local = sample(active_nodes)
      let new_node_id = add_node(top)
      let new_local_node_id = nn.meta_nodes.len
      nn.meta_nodes.add(new_hidden_node(new_node_id))

      var new_edges_to_add: seq[(int, int, int)] = @[]
      for i in 0 ..< meta_edges_len:
        let meta_edge = nn.meta_edges[i]
        if meta_edge.disabled: continue

        let from_local = meta_edge.local_from_node_id
        let to_local = meta_edge.local_to_node_id

        if to_local == chosen_local:
          let from_global = nn.meta_nodes[from_local].node_id
          let edge_id = if not top.conn_index.has_key((from_global, new_node_id)):
            add_edge(top, from_global, new_node_id)
          else:
            top.conn_index[(from_global, new_node_id)]
          new_edges_to_add.add((edge_id, from_local, new_local_node_id))
        elif from_local == chosen_local:
          if not nn.meta_nodes[to_local].frozen:
            let to_global = nn.meta_nodes[to_local].node_id
            let edge_id = if not top.conn_index.has_key((new_node_id, to_global)):
              add_edge(top, new_node_id, to_global)
            else:
              top.conn_index[(new_node_id, to_global)]
            new_edges_to_add.add((edge_id, new_local_node_id, to_local))

      for i in 0 ..< new_edges_to_add.len:
        let edge_info = new_edges_to_add[i]
        add_meta_edge(edge_info[0], edge_info[1], edge_info[2], random_normal())
      mutated = true

  template do_grow_edge() =
    var active_edges: seq[int] = @[]
    for i in 0 ..< meta_edges_len:
      if not nn.meta_edges[i].disabled and not nn.meta_edges[i].frozen:
        active_edges.add(i)

    if active_edges.len > 0:
      let chosen_edge = sample(active_edges)
      nn.meta_edges[chosen_edge].disabled = true

      let meta_edge = nn.meta_edges[chosen_edge]
      let edge = top.edges[meta_edge.edge_id]

      var node_id, first_edge_id, second_edge_id: int
      if top.edge_splits.has_key(meta_edge.edge_id):
        let (split_node_id, split_first_edge_id, split_second_edge_id) = top.edge_splits[meta_edge.edge_id]
        node_id = split_node_id
        first_edge_id = split_first_edge_id
        second_edge_id = split_second_edge_id
      else:
        node_id = add_node(top)
        first_edge_id = add_edge(top, edge.from_node_id, node_id)
        second_edge_id = add_edge(top, node_id, edge.to_node_id)
        top.edge_splits[meta_edge.edge_id] = (node_id, first_edge_id, second_edge_id)

      var new_local_node_id = find_local_node_id(nn, node_id)
      if new_local_node_id >= 0:
        if not nn.meta_nodes[new_local_node_id].frozen:
          nn.meta_nodes[new_local_node_id].disabled = false
      else:
        new_local_node_id = nn.meta_nodes.len
        nn.meta_nodes.add(new_hidden_node(node_id))

      let local_first_edge = find_local_edge_id(nn, first_edge_id)
      if local_first_edge >= 0:
        if not nn.meta_edges[local_first_edge].frozen:
          nn.meta_edges[local_first_edge].disabled = false
          nn.meta_edges[local_first_edge].weight = 1.0
      else:
        add_meta_edge(first_edge_id, meta_edge.local_from_node_id, new_local_node_id, 1.0)

      let local_second_edge = find_local_edge_id(nn, second_edge_id)
      if local_second_edge >= 0:
        if not nn.meta_edges[local_second_edge].frozen:
          nn.meta_edges[local_second_edge].disabled = false
          nn.meta_edges[local_second_edge].weight = meta_edge.weight
      else:
        add_meta_edge(second_edge_id, new_local_node_id, meta_edge.local_to_node_id, meta_edge.weight)
      mutated = true

  template do_add_novel_edge() =
    var active_from_nodes: seq[int] = @[]
    var active_to_nodes: seq[int] = @[]
    for i in 0 ..< nn.meta_nodes.len:
      let node = nn.meta_nodes[i]
      if node.disabled: continue

      let node_type = top.nodes[node.node_id].`type`
      if node_type != NodeType.output or node.frozen:
        active_from_nodes.add(i)
      if node_type != NodeType.input and not node.frozen:
        active_to_nodes.add(i)

    if active_from_nodes.len > 0 and active_to_nodes.len > 0:
      let from_local = sample(active_from_nodes)
      var to_local = sample(active_to_nodes)

      # resample a few times to avoid self loops
      var attempts = 0
      while from_local == to_local and attempts < 10:
        to_local = sample(active_to_nodes)
        attempts += 1

      if from_local != to_local:
        let from_global = nn.meta_nodes[from_local].node_id
        let to_global = nn.meta_nodes[to_local].node_id
        let connection = (from_global, to_global)
        let edge_id = if not top.conn_index.has_key(connection):
          add_edge(top, from_global, to_global)
        else:
          top.conn_index[connection]

        let local_edge = find_local_edge_id(nn, edge_id)
        if local_edge < 0:
          add_meta_edge(edge_id, from_local, to_local, random_normal())
        elif not nn.meta_edges[local_edge].frozen:
          nn.meta_edges[local_edge].disabled = false

        mutated = true

  # structural mutations - fired ONCE per individual, not per node/edge

  if satisfy_prob(hparams.grow_node_prob):
    do_grow_node()

  if satisfy_prob(hparams.grow_edge_prob):
    do_grow_edge()

  if satisfy_prob(hparams.add_novel_edge_prob):
    do_add_novel_edge()

  if force and not mutated:
    let r = rand(2)
    if r == 0:
      do_grow_edge()
      if not mutated: do_add_novel_edge()
      if not mutated: do_grow_node()
    elif r == 1:
      do_add_novel_edge()
      if not mutated: do_grow_edge()
      if not mutated: do_grow_node()
    else:
      do_grow_node()
      if not mutated: do_grow_edge()
      if not mutated: do_add_novel_edge()

  if mutated:
    nn.cached_exec_trace = none(ExecTrace)

  return mutated
  {.pop.}

proc mutate(
  top: Topology,
  nn_id: int,
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.gcsafe exportpy.} =
  mutate_weights(top, nn_id, mutation_hyper_params, sync_trace = false)
  let struct_mutated = mutate_structure(top, nn_id, mutation_hyper_params)
  if not struct_mutated:
    sync_exec_trace_weights(top.population[nn_id])

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
    let num_islands = top.num_islands
    let pop_size = top.population.len
    let island_pop_size = pop_size div num_islands

    assert hyper_params.num_preserve_elites < island_pop_size
    for i in 0 ..< num_islands:
      let offset = i * island_pop_size
      for nn_id in (offset + hyper_params.num_preserve_elites) ..< (offset + island_pop_size):
        mutate(top, nn_id, mutation_hyper_params)

    set_population_exec_trace(top_id)

type
  CrossoverScratch = object
    parent1_node_index: seq[int]
    parent2_node_index: seq[int]
    child_node_index: seq[int]
    parent1_edge_index: seq[int]
    parent2_edge_index: seq[int]
    joint_nodes: seq[int]
    disjoint_nodes: seq[int]
    joint_edges: seq[int]
    disjoint_edges: seq[int]

var scratch_crossover {.threadvar.}: CrossoverScratch

proc ensure_index_table(indices: var seq[int], req_len: int) {.inline.} =
  # new slots start at -1, meaning "absent"
  let old_len = indices.len
  if old_len >= req_len: return

  indices.set_len(req_len)
  for i in old_len ..< req_len:
    indices[i] = -1

proc crossover(
  top: Topology,
  first_parent_nn_id: int,
  second_parent_nn_id: int,
  first_parent_fitness: float32,
  second_parent_fitness: float32,
  crossover_hyper_params: Option[CrossoverHyperParams] = CrossoverHyperParams.none
): NeuralNetwork {.exportpy.} =
  {.push checks: off.}
  let hyper_params = crossover_hyper_params.get(top.crossover_hyper_params)

  let parent1 = top.population[first_parent_nn_id]
  let parent2 = top.population[second_parent_nn_id]

  let fitter_parent =
    if first_parent_fitness > second_parent_fitness: parent1
    elif second_parent_fitness > first_parent_fitness: parent2
    elif coin_flip(): parent1
    else: parent2

  let max_nodes = top.nodes.len
  let max_edges = top.edges.len

  ensure_index_table(scratch_crossover.parent1_node_index, max_nodes)
  ensure_index_table(scratch_crossover.parent2_node_index, max_nodes)
  ensure_index_table(scratch_crossover.child_node_index, max_nodes)
  ensure_index_table(scratch_crossover.parent1_edge_index, max_edges)
  ensure_index_table(scratch_crossover.parent2_edge_index, max_edges)

  let p1_node_index = scratch_crossover.parent1_node_index.raw_ptr
  let p2_node_index = scratch_crossover.parent2_node_index.raw_ptr
  let child_node_index = scratch_crossover.child_node_index.raw_ptr
  let p1_edge_index = scratch_crossover.parent1_edge_index.raw_ptr
  let p2_edge_index = scratch_crossover.parent2_edge_index.raw_ptr

  let p1_nodes_len = parent1.meta_nodes.len
  let p2_nodes_len = parent2.meta_nodes.len
  let p1_edges_len = parent1.meta_edges.len
  let p2_edges_len = parent2.meta_edges.len

  for i in 0 ..< p1_nodes_len:
    p1_node_index[parent1.meta_nodes[i].node_id] = i
  for i in 0 ..< p2_nodes_len:
    p2_node_index[parent2.meta_nodes[i].node_id] = i

  for i in 0 ..< p1_edges_len:
    p1_edge_index[parent1.meta_edges[i].edge_id] = i
  for i in 0 ..< p2_edges_len:
    p2_edge_index[parent2.meta_edges[i].edge_id] = i

  # shared genes, i.e. same innovation id in both parents

  scratch_crossover.joint_nodes.set_len(0)
  for node in parent1.meta_nodes:
    if p2_node_index[node.node_id] >= 0:
      scratch_crossover.joint_nodes.add(node.node_id)
  scratch_crossover.joint_nodes.sort()

  scratch_crossover.joint_edges.set_len(0)
  for edge in parent1.meta_edges:
    if p2_edge_index[edge.edge_id] >= 0:
      scratch_crossover.joint_edges.add(edge.edge_id)
  scratch_crossover.joint_edges.sort()

  # excess genes are inherited from the fitter parent

  scratch_crossover.disjoint_nodes.set_len(0)
  scratch_crossover.disjoint_edges.set_len(0)

  var disjoint_from_p1 = true
  if satisfy_prob(hyper_params.prob_inherit_all_excess_genes):
    disjoint_from_p1 = fitter_parent == parent1

    let disjoint_parent = if disjoint_from_p1: parent1 else: parent2
    let other_node_index = if disjoint_from_p1: p2_node_index else: p1_node_index
    let other_edge_index = if disjoint_from_p1: p2_edge_index else: p1_edge_index

    for node in disjoint_parent.meta_nodes:
      if other_node_index[node.node_id] < 0:
        scratch_crossover.disjoint_nodes.add(node.node_id)

    for edge in disjoint_parent.meta_edges:
      if other_edge_index[edge.edge_id] < 0:
        scratch_crossover.disjoint_edges.add(edge.edge_id)

    scratch_crossover.disjoint_nodes.sort()
    scratch_crossover.disjoint_edges.sort()

  var child_nodes = new_seq_of_cap[MetaNode](scratch_crossover.joint_nodes.len + scratch_crossover.disjoint_nodes.len)
  var child_edges = new_seq_of_cap[MetaEdge](scratch_crossover.joint_edges.len + scratch_crossover.disjoint_edges.len)

  # joint nodes

  for node_id in scratch_crossover.joint_nodes:
    let parent1_node = parent1.meta_nodes[p1_node_index[node_id]]
    let parent2_node = parent2.meta_nodes[p2_node_index[node_id]]
    let chosen_node = if coin_flip(): parent1_node else: parent2_node

    let new_node = MetaNode()
    new_node[] = chosen_node[]

    if not new_node.frozen and (parent1_node.disabled or parent2_node.disabled):
      new_node.disabled = satisfy_prob(hyper_params.prob_child_disabled_given_parent_cond)

    if new_node.disabled and new_node.can_disable and not new_node.frozen:
      if satisfy_prob(hyper_params.prob_remove_disabled_node):
        continue

    child_node_index[node_id] = child_nodes.len
    child_nodes.add(new_node)

  # disjoint and excess nodes

  for node_id in scratch_crossover.disjoint_nodes:
    let source_node =
      if disjoint_from_p1: parent1.meta_nodes[p1_node_index[node_id]]
      else: parent2.meta_nodes[p2_node_index[node_id]]

    let new_node = MetaNode()
    new_node[] = source_node[]

    if new_node.disabled and new_node.can_disable and not new_node.frozen:
      if satisfy_prob(hyper_params.prob_remove_disabled_node):
        continue

    child_node_index[node_id] = child_nodes.len
    child_nodes.add(new_node)

  template add_child_edge(candidate: MetaEdge) =
    let global_edge = top.edges[candidate.edge_id]
    let local_from = child_node_index[global_edge.from_node_id]
    let local_to = child_node_index[global_edge.to_node_id]

    if local_from >= 0 and local_to >= 0:
      var child_edge = candidate
      child_edge.local_from_node_id = local_from
      child_edge.local_to_node_id = local_to
      child_edges.add(child_edge)

  # joint edges

  for edge_id in scratch_crossover.joint_edges:
    let parent1_edge = parent1.meta_edges[p1_edge_index[edge_id]]
    let parent2_edge = parent2.meta_edges[p2_edge_index[edge_id]]
    var new_edge = if coin_flip(): parent1_edge else: parent2_edge

    if not new_edge.frozen and (parent1_edge.disabled or parent2_edge.disabled):
      new_edge.disabled = satisfy_prob(hyper_params.prob_child_disabled_given_parent_cond)

    add_child_edge(new_edge)

  # disjoint and excess edges

  for edge_id in scratch_crossover.disjoint_edges:
    let new_edge =
      if disjoint_from_p1: parent1.meta_edges[p1_edge_index[edge_id]]
      else: parent2.meta_edges[p2_edge_index[edge_id]]

    add_child_edge(new_edge)

  # clear scratch tables for the next crossover

  for node in parent1.meta_nodes:
    p1_node_index[node.node_id] = -1
    child_node_index[node.node_id] = -1
  for node in parent2.meta_nodes:
    p2_node_index[node.node_id] = -1
    child_node_index[node.node_id] = -1
  for edge in parent1.meta_edges:
    p1_edge_index[edge.edge_id] = -1
  for edge in parent2.meta_edges:
    p2_edge_index[edge.edge_id] = -1

  return NeuralNetwork(
    num_inputs: top.num_inputs,
    num_outputs: top.num_outputs,
    num_hiddens: top.num_hiddens,
    meta_nodes: child_nodes,
    meta_edges: child_edges,
    mutation_step_size: fitter_parent.mutation_step_size
  )
  {.pop.}

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
      for i in 0 ..< couples.len:
        let nn_id = top.curr_pop_size + i
        if nn_id < top.pop_size:
          crossover_one_couple_and_add_to_population(top, nn_id, couples[i], crossover_hyper_params)

    top.curr_pop_size = top.pop_size

# migration and island reset

proc clone*(nn: NeuralNetwork): NeuralNetwork =
  {.push checks: off.}
  let n_nodes = nn.meta_nodes.len
  var cloned_nodes = new_seq[MetaNode](n_nodes)
  for i in 0 ..< n_nodes:
    let new_node = MetaNode()
    new_node[] = nn.meta_nodes[i][]
    cloned_nodes[i] = new_node

  let cloned_edges = nn.meta_edges

  let cloned_trace = if nn.cached_exec_trace.is_some:
    some(clone_exec_trace(nn.cached_exec_trace.get))
  else:
    none(ExecTrace)

  return NeuralNetwork(
    id: nn.id,
    topology_id: nn.topology_id,
    num_inputs: nn.num_inputs,
    num_outputs: nn.num_outputs,
    num_hiddens: nn.num_hiddens,
    meta_nodes: cloned_nodes,
    meta_edges: cloned_edges,
    cached_exec_trace: cloned_trace,
    mutation_step_size: nn.mutation_step_size,
  )
  {.pop.}

proc clone_nn*(top_id: int, nn_id: int): NeuralNetwork {.exportpy.} =
  let top = topologies[top_id]
  return top.population[nn_id].clone()

proc set_nn*(top_id: int, nn_id: int, nn: NeuralNetwork) {.exportpy.} =
  let top = topologies[top_id]
  top.population[nn_id] = nn

proc clone_nn_obj*(nn: NeuralNetwork): NeuralNetwork {.exportpy.} =
  return nn.clone()

proc mutate_selected*(
  all_top_ids: seq[int],
  nn_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  for top_id in all_top_ids:
    let top = topologies[top_id]
    for nn_id in nn_ids:
      mutate(top, nn_id, mutation_hyper_params)
    set_population_exec_trace(top_id)

proc mutate_selected_weights*(
  all_top_ids: seq[int],
  nn_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  for top_id in all_top_ids:
    let top = topologies[top_id]
    for nn_id in nn_ids:
      mutate_weights(top, nn_id, mutation_hyper_params)
    set_population_exec_trace(top_id)

proc mutate_selected_structural*(
  all_top_ids: seq[int],
  nn_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  for top_id in all_top_ids:
    let top = topologies[top_id]
    for nn_id in nn_ids:
      discard mutate_structure(top, nn_id, mutation_hyper_params)
    set_population_exec_trace(top_id)

proc mutate_selected_structural_forced*(
  all_top_ids: seq[int],
  nn_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  for top_id in all_top_ids:
    let top = topologies[top_id]
    for nn_id in nn_ids:
      discard mutate_structure(top, nn_id, mutation_hyper_params, force = true)
    set_population_exec_trace(top_id)

proc mutate_survivors*(
  all_top_ids: seq[int],
  mutation_hyper_params: Option[MutationHyperParams] = MutationHyperParams.none
) {.exportpy.} =
  for top_id in all_top_ids:
    let top = topologies[top_id]
    let hyper_params = mutation_hyper_params.get(top.mutation_hyper_params)
    let sel_hparams = top.selection_hyper_params
    let num_islands = top.num_islands
    let pop_size = top.population.len
    let island_pop_size = pop_size div num_islands
    let num_selected_per_island = max(2, (sel_hparams.frac_natural_selected * island_pop_size.float).int)

    for i in 0 ..< num_islands:
      let offset = i * island_pop_size
      let start_idx = offset + hyper_params.num_preserve_elites
      let end_idx = offset + num_selected_per_island
      for nn_id in start_idx ..< end_idx:
        mutate(top, nn_id, mutation_hyper_params)

    set_population_exec_trace(top_id)

proc is_hidden_node(top: Topology, node_id: int): bool {.inline.} =
  node_id < top.nodes.len and top.nodes[node_id].`type` == NodeType.hidden

proc set_frozen*(
  top_id: int,
  frozen: bool = true,
  except_output_indices: seq[int] = @[]
) {.exportpy.} =
  let top = topologies[top_id]
  var exempt_node_ids = init_hash_set[int]()
  for output_index in except_output_indices:
    exempt_node_ids.incl(top.num_inputs + output_index)

  for nn in top.population:
    for node in nn.meta_nodes:
      if node.node_id notin exempt_node_ids:
        let is_hidden = top.is_hidden_node(node.node_id)
        node.frozen = frozen
        node.can_disable = (not frozen) and is_hidden
        node.can_change_activation = (not frozen) and is_hidden
        node.can_change_aggregation = (not frozen) and is_hidden

    for edge in nn.meta_edges.mitems:
      if top.edges[edge.edge_id].to_node_id notin exempt_node_ids:
        edge.frozen = frozen
        edge.can_disable = not frozen

proc set_frozen_subset*(
  top_id: int,
  node_ids: seq[int],
  edge_ids: seq[int],
  frozen: bool = true
) {.exportpy.} =
  let top = topologies[top_id]
  let node_set = node_ids.to_hash_set
  let edge_set = edge_ids.to_hash_set

  for nn in top.population:
    for node in nn.meta_nodes:
      if node.node_id in node_set:
        let is_hidden = top.is_hidden_node(node.node_id)
        node.frozen = frozen
        node.can_disable = (not frozen) and is_hidden
        node.can_change_activation = (not frozen) and is_hidden
        node.can_change_aggregation = (not frozen) and is_hidden

    for edge in nn.meta_edges.mitems:
      if edge.edge_id in edge_set:
        edge.frozen = frozen
        edge.can_disable = not frozen

proc freeze_population*(top_id: int, except_output_indices: seq[int] = @[]) {.exportpy.} =
  set_frozen(top_id, true, except_output_indices)

proc unfreeze_population*(top_id: int) {.exportpy.} =
  set_frozen(top_id, false)

proc freeze_subset*(top_id: int, node_ids: seq[int], edge_ids: seq[int]) {.exportpy.} =
  set_frozen_subset(top_id, node_ids, edge_ids, true)

proc unfreeze_subset*(top_id: int, node_ids: seq[int], edge_ids: seq[int]) {.exportpy.} =
  set_frozen_subset(top_id, node_ids, edge_ids, false)

proc get_frozen_stats*(
  top_id: int
): tuple[frozen_nodes: int, total_nodes: int, frozen_edges: int, total_edges: int] {.exportpy.} =
  let top = topologies[top_id]
  if top.population.len == 0:
    return (0, 0, 0, 0)

  let nn = top.population[0]

  var frozen_nodes = 0
  for node in nn.meta_nodes:
    if node.frozen: inc frozen_nodes

  var frozen_edges = 0
  for edge in nn.meta_edges:
    if edge.frozen: inc frozen_edges

  return (frozen_nodes, nn.meta_nodes.len, frozen_edges, nn.meta_edges.len)

proc migrate_islands*(
  all_top_ids: seq[int],
  num_migrants: int
) {.exportpy.} =
  if all_top_ids.len == 0: return

  let first_top = topologies[all_top_ids[0]]
  let num_islands = first_top.num_islands
  if num_islands <= 1: return

  let pop_size = first_top.pop_size
  let island_pop_size = pop_size div num_islands
  if num_migrants >= island_pop_size: return

  for top_id in all_top_ids:
    let top = topologies[top_id]
    var new_population = new_seq[NeuralNetwork](pop_size)

    for island_id in 0 ..< num_islands:
      let offset = island_id * island_pop_size
      let previous_island = (island_id + num_islands - 1) mod num_islands
      let previous_offset = previous_island * island_pop_size

      for i in 0 ..< island_pop_size:
        new_population[offset + i] = top.population[offset + i]

      for i in 0 ..< num_migrants:
        new_population[offset + island_pop_size - 1 - i] = top.population[previous_offset + i].clone()

    for i in (num_islands * island_pop_size) ..< pop_size:
      new_population[i] = top.population[i]

    top.population = new_population
    set_population_exec_trace(top_id)

proc reset_top_islands*(
  all_top_ids: seq[int],
  fitnesses: seq[float32],
  num_islands_to_reset: int,
  tournament_size: int
) {.exportpy.} =
  if all_top_ids.len == 0: return

  let first_top = topologies[all_top_ids[0]]
  let num_islands = first_top.num_islands
  let num_reset = min(num_islands_to_reset, num_islands - 1)
  if num_islands <= 1 or num_reset <= 0: return

  let pop_size = first_top.pop_size
  let island_pop_size = pop_size div num_islands

  var island_avgs: seq[(float32, int)] = @[]
  var sorted_island_fits = new_seq[seq[float32]](num_islands)

  for island_id in 0 ..< num_islands:
    let offset = island_id * island_pop_size
    var island_fits = fitnesses[offset ..< offset + island_pop_size]

    var total = 0.0'f32
    for fitness in island_fits:
      total += fitness

    island_avgs.add((total / island_pop_size.float32, island_id))

    island_fits.sort(order = SortOrder.Descending)
    sorted_island_fits[island_id] = island_fits

  island_avgs.sort()

  let reset_island_ids = island_avgs[0 ..< num_reset].map(x => x[1])
  let survivor_island_ids = island_avgs[num_reset .. ^1].map(x => x[1])

  var global_survivors: seq[(int, float32)] = @[]
  let num_selected_per_island = max(
    2,
    (first_top.selection_hyper_params.frac_natural_selected * island_pop_size.float).int
  )

  for island_id in survivor_island_ids:
    let offset = island_id * island_pop_size
    for i in 0 ..< num_selected_per_island:
      global_survivors.add((offset + i, sorted_island_fits[island_id][i]))

  for island_id in reset_island_ids:
    let offset = island_id * island_pop_size

    for nn_id in offset ..< offset + island_pop_size:
      var
        parent1, parent2: int = -1
        fitness1, fitness2: float32 = low(float32)

      for _ in 0 ..< tournament_size:
        let (index, fitness) = global_survivors[rand(global_survivors.len - 1)]
        if fitness > fitness1:
          parent2 = parent1
          fitness2 = fitness1
          parent1 = index
          fitness1 = fitness
        elif fitness > fitness2:
          parent2 = index
          fitness2 = fitness

      if parent1 < 0:
        parent1 = global_survivors[0][0]
        fitness1 = global_survivors[0][1]
      if parent2 < 0:
        parent2 = parent1
        fitness2 = fitness1

      for top_id in all_top_ids:
        let top = topologies[top_id]
        top.population[nn_id] = crossover(top, parent1, parent2, fitness1, fitness2)
        mutate(top, nn_id)

  for top_id in all_top_ids:
    set_population_exec_trace(top_id)

# quick test

when is_main_module:

  # neat

  let hyperneat_top_id = add_topology(3, 1, @[16, 16, 16])
  init_population(hyperneat_top_id, 10)
  var top = topologies[hyperneat_top_id]

  assert top.population.len == 10

  for i in 0..<100:
    let (_, _, couples, target_nn_ids) = select_and_tournament(
      @[hyperneat_top_id],
      @[1.0'f32, 2.0, 3.0, 5.0, 4.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    )
    crossover_and_add_to_population(@[hyperneat_top_id], couples, target_nn_ids)
    mutate_all(@[hyperneat_top_id])

  save_json_to_file(hyperneat_top_id, "./population.json")
