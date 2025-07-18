import nimpy

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
  locks,
  algorithm
]

import bitvector

import arraymancer

import jsony

import weave

Weave.init()

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

proc random_normal(): float =
  # box-muller for random normal
  let
    u1 = rand(1.0)
    u2 = rand(1.0)

  return sqrt(-2 * ln(u1)) * cos(2 * PI * u2)

# normalization

proc min_max_norm(
  fitnesses: seq[float]
): seq[float] =

  let
    min = fitnesses.min
    max = fitnesses.max

  let divisor = max - min

  if divisor == 0.0:
    return fitnesses

  return fitnesses.map((fitness) => (fitness - min) / divisor)

# activation functions

proc sigmoid(x: float): float =
  return (1.0 / (1.0 + exp(-x)))

proc relu(x: float): float =
  return max(x, 0.0)

proc gauss(x: float): float =
  return exp(-pow(x, 2))

proc identity(x: float): float =
  return x

proc elu(x: float): float =
  if x >= 0.0:
    result = x
  else:
    result = exp(x) - 1.0

proc clamp_one(x: float): float =
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
    weight: float
    from_node_id: int

  NodeUpdate = tuple
    to_node_id: int
    activation_id: int
    bias: float
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
    bias: float = 0.0

  MetaEdge = ref object
    topology_id: int
    edge_id: int
    disabled: bool
    weight: float
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

  Topology = ref object
    id: int
    lock: Lock

    nodes: seq[Node] = @[] # nodes will be always arrange [input] [output] [hiddens]
    edges: seq[Edge] = @[]

    num_inputs: int
    num_outputs: int
    num_hiddens: seq[int] = @[]

    nn_id: int = 0
    node_innovation_id: int = 0
    edge_innovation_id: int = 0

    pop_size: int = 0
    population: seq[NeuralNetwork] = @[]

  # crossover related

  ParentAndFitness = tuple
    index: int
    fitness: float

  Couple = tuple
    parent1: ParentAndFitness
    parent2: ParentAndFitness

  Couples = seq[Couple]

  TopologyInfo = object
    population_size: int
    total_innovated_nodes: int
    total_innovated_edges: int

# globals

var topologies = init_table[int, Topology]()
var topology_id = 0

# helper accessors

proc get_parent_node(meta_node: MetaNode): Node =
  let top = topologies[meta_node.topology_id]

  for node in top.nodes:
    if node.id == meta_node.node_id:
      result = node

proc get_parent_edge(meta_edge: MetaEdge): Edge =
  let top = topologies[meta_edge.topology_id]

  for edge in top.edges:
    if edge.id == meta_edge.edge_id:
      result = edge

proc get_topology_info(
  top_id: int
): TopologyInfo {.exportpy.} =

  let top = topologies[top_id]

  result.population_size = top.population.len
  result.total_innovated_nodes = top.node_innovation_id + 1
  result.total_innovated_edges = top.edge_innovation_id + 1

# functions

proc add_node(topology_id: int, node_type: NodeType = hidden): int
proc add_edge(topology_id: int, from_node_id: int, to_node_id: int): int

proc activate(act: Activation, input: Tensor[float]): Tensor[float] {.gcsafe.}
proc activate(act: Activation, input: float): float {.gcsafe.}
proc activate(node: MetaNode, input: Tensor[float]): Tensor[float] {.gcsafe.}

proc rand_activation(): Activation

proc add_topology(
  num_inputs: int,
  num_outputs: int,
  num_hiddens: seq[int]
): int {.exportpy.} =

  let topology = Topology(
    id: topology_id,
    num_inputs: num_inputs,
    num_outputs: num_outputs,
    num_hiddens: num_hiddens
  )

  topologies[topology_id] = topology

  topology_id += 1

  # create input and output nodes

  let
    input_node_ids = (0..<num_inputs).to_seq
    output_node_ids = (num_inputs..<(num_inputs + num_outputs)).to_seq

  for _ in 0..<num_inputs:
    discard add_node(topology.id, NodeType.input)

  for _ in 0..<num_outputs:
    discard add_node(topology.id, NodeType.output)

  # create edges

  for input_id in input_node_ids:
    for output_id in output_node_ids:
      discard add_edge(topology.id, input_id, output_id)

  # initial pool of hidden nodes and edges, all disabled for new neural networks at start

  var hidden_node_ids: seq[int] = @[]

  for num_hidden_layer in num_hiddens:
    var layer_hidden_ids: seq[int] = @[]

    for _ in 0..<num_hidden_layer:
      layer_hidden_ids.add(add_node(topology.id, NodeType.hidden))

    hidden_node_ids.add(layer_hidden_ids)

  var all_ids: seq[seq[int]] = @[input_node_ids] & hidden_node_ids & @[output_node_ids]

  for layer_index, from_layer_ids in all_ids[0..^2]:

    let to_layer_ids = all_ids[layer_index + 1]

    for from_id in from_layer_ids:
      for to_id in to_layer_ids:
        discard add_edge(topology.id, from_id, to_id)

  # return id

  return topology.id

proc remove_topology(topology_id: int) {.exportpy.} =
  topologies.del(topology_id)

proc init_top_lock(topology_id: int) {.exportpy.} =
  let top = topologies[topology_id]
  top.lock.init_lock()

proc deinit_top_lock(topology_id: int) {.exportpy.} =
  let top = topologies[topology_id]
  top.lock.deinit_lock()

proc add_node(
  topology_id: int,
  node_type: NodeType = hidden
): int {.exportpy.} =

  let top = topologies[topology_id]

  # create node, increment primary key, and add to global nodes

  let node = Node(id: top.node_innovation_id, topology_id: top.id)
  top.nodes.add(node)

  top.node_innovation_id += 1
  return node.id

proc add_edge(
  topology_id: int,
  from_node_id: int,
  to_node_id: int
): int {.exportpy.} =

  let top = topologies[topology_id]

  # validate node id

  let max_node_id = top.nodes.len
  assert(0 <= from_node_id  and from_node_id < max_node_id)
  assert(0 <= to_node_id and to_node_id < max_node_id)

  # create edge, increment primary key and add to global edges

  let edge = Edge(
    id: top.edge_innovation_id,
    topology_id: top.id,
    from_node_id: from_node_id,
    to_node_id: to_node_id
  )

  top.edges.add(edge)

  top.edge_innovation_id += 1
  return edge.id

# main evolutionary functions

# population functions

proc init_nn(
  top_id: int,
  sparsity: float = 0.05
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
      activation: tanh
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
      can_change_activation: true,
      activation: rand_activation()
    )

    node_index[node.id] = nn.meta_nodes.len
    nn.meta_nodes.add(meta_node)

  # create edges - start off with only fully connected from inputs to outputs

  for index, edge in top.edges[0..<(top.num_inputs * top.num_outputs)]:

      let meta_edge = MetaEdge(
        topology_id: top.id,
        edge_id: edge.id,
        disabled: false,
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

  top.population.add(nn)

proc init_population(
  top_id: int,
  pop_size: range[1..int.high],
) {.exportpy.} =

  let top = topologies[top_id]
  assert top.pop_size == 0

  top.pop_size = pop_size

  for _ in 0..<pop_size:
    init_nn(top_id)

# forward

proc evaluate_nn(
  top_id: int,
  nn_id: int,
  seq_inputs: seq[seq[float]]
): seq[seq[float]] {.exportpy.} =

  let inputs = seq_inputs.map(seq_float => seq_float.to_tensor)

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  assert top.num_inputs == inputs.len

  let one_input = inputs[0]
  let one_input_shape = one_input.shape

  let num_nodes = nn.meta_nodes.len

  var visited = new_bit_vector[uint](num_nodes)
  var finished = new_bit_vector[uint](num_nodes)
  var values = new_seq[Tensor[float]](num_nodes)

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
  ): Tensor[float] =

    if finished[index] == 1:
      return values[index]

    let meta_node = nn.meta_nodes[index]

    # start with bias

    var next_visited = visited
    next_visited[index] = 1

    var node_value = zeros[float](one_input_shape) +. meta_node.bias

    # find all edges

    var input_node_index_and_weight: seq[(float, int)] = @[] # omit visited

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
  top_id: int,
  nn_id: int,
): ExecTrace {.exportpy.} =

  var trace = new_seq[NodeUpdate]()

  let top = topologies[top_id]
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
      0.0,
      @[(1.0, node_index)]
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

    var input_node_index_and_weight: seq[(float, int)] = @[] # omit visited

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

    var multiplies: seq[(float, int)] = @[]

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

proc evaluate_nn_with_trace(
  trace_with_meta_info: ExecTrace,
  inputs: seq[seq[float]]
): seq[seq[float]] {.gcsafe exportpy.} =

  var (meta_info, trace) = trace_with_meta_info
  let (num_inputs, num_outputs, num_nodes) = meta_info

  let seq_inputs_tensor = inputs.map(seq_float => seq_float.to_tensor)

  let one_input = seq_inputs_tensor[0]
  let one_input_shape = one_input.shape

  var values = new_seq_with(num_nodes, zeros[float](one_input_shape))

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
  inputs: seq[float],
  use_exec_cache: bool = false
): seq[float] {.exportpy.} =

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
  trace: ExecTrace,
  buffer_input: ptr UncheckedArray[float],
  buffer_output: ptr UncheckedArray[float]
) {.gcsafe.} =

  let num_inputs = trace.node_info.num_inputs
  var inputs = new_seq[float](num_inputs)

  for i in 0 ..< num_inputs:
    inputs[i] = buffer_input[i]

  let seq_inputs = inputs.map(input => @[input])

  let seq_output = evaluate_nn_with_trace(trace, seq_inputs)

  let outputs = seq_output.map(output => output[0])

  for i in 0 ..< outputs.len:
    buffer_output[i] = outputs[i]

proc evaluate_population(
  top_id: int,
  inputs: seq[seq[float]]
): seq[seq[float]] {.exportpy.} =

  let top = topologies[top_id]

  assert inputs.len == top.pop_size

  for nn_id, input in inputs:

    let nn = top.population[nn_id]
    assert nn.num_inputs == input.len

    # set the cached graph execution on neural network if not exists (it has been mutated)

    if nn.cached_exec_trace.is_none:
      nn.cached_exec_trace = evaluate_nn_exec_trace(top_id, nn_id).some

  # using weave for multi-threading

  let output = new_seq_with(top.population.len, new_seq[float](top.num_outputs))

  parallel_for nn_id in 0 ..< inputs.len:
    captures: {top, inputs, output}

    let nn = top.population[nn_id]

    # input and output buffers for thread

    let buffer_input = cast[ptr UncheckedArray[float]](inputs[nn_id][0].addr)
    let buffer_output = cast[ptr UncheckedArray[float]](output[nn_id][0].addr)

    # spawn thread

    spawn evaluate_nn_single_with_trace_thread_fn(
      nn.cached_exec_trace.get,
      buffer_input,
      buffer_output
    )

  return output

proc generate_hyper_weights(
  top_id: int,
  nn_id: int,
  shape: seq[int]
): seq[float] {.exportpy.} =

  let top = topologies[top_id]

  assert top.num_inputs == shape.len

  let
    first_axis = shape[0]
    rest_axis = shape[1..^1]

  var coors = linspace(-1.0, 1.0, shape[0])
  coors = coors.reshape(1, first_axis)

  for dim in rest_axis:
    var next_dim_coors = linspace(0.0, 1.0, dim)
    next_dim_coors = next_dim_coors.reshape(1, 1, dim).broadcast(1, coors.shape[1], dim)
    coors = coors.reshape(coors.shape[0], coors.shape[1], 1).broadcast(coors.shape[0], coors.shape[1], dim)
    coors = concat(coors, next_dim_coors, axis = 0)
    coors = coors.reshape(coors.shape[0], coors.shape[1] * coors.shape[2])

  var weights = evaluate_nn(top_id, nn_id, coors.to_seq_2d).to_tensor
  let meta_data = to_metadata(shape)

  return weights.reshape(meta_data).to_seq

proc activate(
  node: MetaNode,
  input: Tensor[float]
): Tensor[float] {.gcsafe.} =

  return activate(node.activation, input)

proc activate(
  act: Activation,
  input: Tensor[float]
): Tensor[float]  {.gcsafe.} =

  return input.map(value => activate(act, value))

proc activate(
  act: Activation,
  input: float
): float {.gcsafe.} =

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
  fitnesses: seq[float],
  num_tournaments: range[1..int.high],
  tournament_size: range[2..int.high]
): Couples {.exportpy.} =

  var gene_ids = arange(fitnesses.len).to_seq()

  for _ in 0..<num_tournaments:

    var
      parent1, parent2: int = -1
      fitness1, fitness2: float = -1e6

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
  fitnesses: seq[float],
  num_selected: range[1..int.high],
  tournament_size: range[2..int.high]
): (
  seq[int],
  seq[float],
  seq[((int, float), (int, float))]
) {.exportpy.} =

  assert top_ids.len > 0

  let one_top_id = top_ids[0]

  let top = topologies[one_top_id]

  let pop_size = top.population.len

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
    top.population = selected_sorted_indices.map(index => top.population[index])

  return (selected_sorted_indices, selected_sorted_fitnesses, parent_indices)

proc mutate(
  top_id: int,
  nn_id: int,
  mutate_prob: Prob = 0.8,
  add_remove_edge_prob: Prob = 0.01,
  add_remove_node_prob: Prob = 0.01,
  change_activation_prob: Prob = 0.01,
  change_edge_weight_prob: Prob = 0.01,
  replace_edge_weight_prob: Prob = 0.25,   # the percentage of time to replace the edge weight wholesale, which they did in the paper in addition to perturbing
  change_node_bias_prob: Prob = 0.051,
  decay_edge_weight_prob: Prob = 0.0,
  decay_node_bias_prob: Prob = 0.0,
  grow_edge_prob: Prob = 0.001,            # this is the mutation introduced in the seminal NEAT paper that takes an existing edge for a CPPN and disables it, replacing it with a new node plus two new edges. the afferent edge is initialized to 1, the efferent inherits same weight as the one disabled. this is something currently neural network frameworks simply cannot do, and what interests me
  grow_node_prob: Prob = 0.0,              # similarly, some follow up research do a variation of the above and split an existing node into two nodes
  perturb_weight_strength: Prob = 0.25,
  perturb_bias_strength: Prob = 0.25,
  decay_factor: float = 0.95
) {.exportpy.} =

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  if not satisfy_prob(mutate_prob):
    return

  var node_index = init_table[int, int]()
  var edge_index = init_table[int, int]()

  # mutating nodes

  for local_node_id, meta_node in nn.meta_nodes:
    node_index[meta_node.node_id] = local_node_id

    # mutating an activation on a node

    if meta_node.can_change_activation and satisfy_prob(change_activation_prob):
      meta_node.activation = rand_activation()

    if meta_node.disabled:
      continue

    if satisfy_prob(change_node_bias_prob):
      meta_node.bias += random_normal() * perturb_bias_strength

    if satisfy_prob(decay_node_bias_prob):
      meta_node.bias *= decay_factor

  # add / remove node

  for node in top.nodes:
    # enabling / disabling an edge

    if not satisfy_prob(add_remove_node_prob):
      continue

    if node.id notin node_index:
      let new_meta_node = MetaNode(
        topology_id: top.id,
        activation: rand_activation(),
        can_disable: false, # only inputs and outputs are locked
        disabled: true
      )

      node_index[node.id] = nn.meta_nodes.len
      nn.meta_nodes.add(new_meta_node)

    let meta_node_id = node_index[node.id]
    let meta_node = nn.meta_nodes[meta_node_id]

    # add or removing an existing node, if valid (input and output nodes are preserved)

    if not meta_node.can_disable:
      continue

    meta_node.disabled = meta_node.disabled xor true
    if not meta_node.disabled:
      meta_node.bias = random_normal()

  # mutating edges

  for meta_edge_index in 0..<nn.meta_edges.len:

    let meta_edge = nn.meta_edges[meta_edge_index]

    edge_index[meta_edge.edge_id] = meta_edge_index

    # changing a weight

    if meta_edge.disabled:
      continue

    if satisfy_prob(change_edge_weight_prob):

      if satisfy_prob(replace_edge_weight_prob):
        meta_edge.weight = random_normal()
      else:
        meta_edge.weight += random_normal() * perturb_weight_strength

    if satisfy_prob(decay_edge_weight_prob):
      meta_edge.weight *= decay_factor

    # maybe splitting an edge
    # this is the novel mutation introduced in the original NEAT paper

    if satisfy_prob(grow_edge_prob):

      # disable the edge

      meta_edge.disabled = true

      # acquire lock

      top.lock.acquire()

      # add a new innovated node

      let node_id = add_node(top_id)

      # add the two innovated edges, with the new node above in between

      let edge = meta_edge.get_parent_edge()

      discard add_edge(top_id, edge.from_node_id, node_id)
      discard add_edge(top_id, node_id, edge.to_node_id)

      # release

      top.lock.release()

      # now add the meta nodes and edges for this particular neural network instantiation

      let meta_node = MetaNode(
        topology_id: top_id,
        node_id: node_id,
        activation: rand_activation()
      )

      let new_local_node_id = nn.meta_nodes.len
      nn.meta_nodes.add(meta_node)

      let meta_edge_incoming = MetaEdge(
        topology_id: top_id,
        local_from_node_id: meta_edge.local_from_node_id,
        local_to_node_id: new_local_node_id,
        weight: 1.0 # they initialize to 1.
      )

      nn.meta_edges.add(meta_edge_incoming)

      let meta_edge_outgoing = MetaEdge(
        topology_id: top_id,
        local_from_node_id: new_local_node_id,
        local_to_node_id: meta_edge.local_to_node_id,
        weight: meta_edge.weight # inherits old weight
      )

      nn.meta_edges.add(meta_edge_outgoing)

  # add / remove edge

  for edge in top.edges:
    # enabling / disabling an edge

    if not satisfy_prob(add_remove_edge_prob):
      continue

    if (
      edge.from_node_id notin node_index or
      edge.to_node_id notin node_index
    ):
      continue

    if edge.id notin edge_index:
      let new_meta_edge = MetaEdge(
        topology_id: top.id,
        local_from_node_id: node_index[edge.from_node_id],
        local_to_node_id: node_index[edge.to_node_id],
        disabled: true
      )

      edge_index[edge.id] = nn.meta_edges.len
      nn.meta_edges.add(new_meta_edge)

    let meta_edge_id = edge_index[edge.id]
    let meta_edge = nn.meta_edges[meta_edge_id]

    meta_edge.disabled = meta_edge.disabled xor true
    if not meta_edge.disabled:
      meta_edge.weight = random_normal()

  # just remove cached trace for now
  # properly detect change in future

  nn.cached_exec_trace = none(ExecTrace)

proc crossover(
  top_id: int,
  first_parent_nn_id: int,
  second_parent_nn_id: int,
  first_parent_fitness: float,
  second_parent_fitness: float,
  fitness_diff_is_same: float = 0.0
): NeuralNetwork {.exportpy.} =

  let top = topologies[top_id]

  # parents

  let parent1 = top.population[first_parent_nn_id]
  let parent2 = top.population[second_parent_nn_id]

  # absolute fitness difference

  let fitness_difference = abs(first_parent_fitness - second_parent_fitness)

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

  let parent1_edges_index = index_meta_edges_by_global_id(parent2.meta_edges)
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

  if fitness_difference <= fitness_diff_is_same:
    joint_node_ids &= (parent1_node_set * parent2_node_set).to_seq
    joint_edge_ids &= (parent1_edge_set * parent2_edge_set).to_seq

  elif first_parent_fitness < second_parent_fitness:
    disjoint_nodes_index = parent2_nodes_index
    disjoint_edges_index = parent2_edges_index
    disjoint_node_ids = (parent2_node_set - parent1_node_set).to_seq
    disjoint_edge_ids = (parent2_edge_set - parent1_edge_set).to_seq

  elif second_parent_fitness > first_parent_fitness:
    disjoint_nodes_index = parent1_nodes_index
    disjoint_edges_index = parent1_edges_index
    disjoint_node_ids = (parent1_node_set - parent2_node_set).to_seq
    disjoint_edge_ids = (parent1_edge_set - parent2_edge_set).to_seq

  # new child node index

  var child_node_index = init_table[int, int]()

  # handle joint nodes

  for node_id in joint_node_ids:

    let rand_node = if coin_flip():
      parent1_nodes_index[node_id]
    else:
      parent2_nodes_index[node_id]

    let new_node = MetaNode()
    new_node[] = rand_node[]

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

    let rand_edge = if coin_flip():
      parent1_edges_index[edge_id]
    else:
      parent2_edges_index[edge_id]

    let new_edge = MetaEdge()
    new_edge[] = rand_edge[]

    let edge = new_edge.get_parent_edge()

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

    let edge = new_edge.get_parent_edge()

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
  top_id: int,
  couple: Couple,
  fitness_diff_is_same: float = 0.0
) {.exportpy.} =

  let top = topologies[top_id]

  let (parent1_info, parent2_info) = couple

  let (parent1, fitness1) = parent1_info
  let (parent2, fitness2) = parent2_info

  let child = crossover(top_id, parent1, parent2, fitness1, fitness2, fitness_diff_is_same)

  with_lock(top.lock):
    top.population.add(child)

proc crossover_and_add_to_population(
  top_id: int,
  couples: Couples,
  fitness_diff_is_same: float = 0.0
) {.exportpy.} =

  let top = topologies[top_id]

  for couple in couples:
    let (parent1_info, parent2_info) = couple

    let (parent1, fitness1) = parent1_info
    let (parent2, fitness2) = parent2_info

    let child = crossover(top_id, parent1, parent2, fitness1, fitness2, fitness_diff_is_same)

    top.population.add(child)

# quick test

when is_main_module:

  # hyperneat

  let hyperneat_top_id = add_topology(3, 1, @[16, 16, 16])
  init_nn(hyperneat_top_id)
  init_population(hyperneat_top_id, 10)

  let output1 = evaluate_nn_single(hyperneat_top_id, 0, @[2.0, 3.0, 5.0])
  let trace = evaluate_nn_exec_trace(hyperneat_top_id, 0)
  let output2 = evaluate_nn_single(hyperneat_top_id, 0, @[2.0, 3.0, 5.0], use_exec_cache = true)

  assert output1 == output2

  benchmark("non cached", 100):
    discard evaluate_nn_single(hyperneat_top_id, 0, @[2.0, 3.0, 5.0])

  benchmark("cached", 100):
    discard evaluate_nn_single(hyperneat_top_id, 0, @[2.0, 3.0, 5.0], use_exec_cache = true)

  discard crossover(hyperneat_top_id, 0, 1, 0.5, 0.3)

  mutate(hyperneat_top_id, nn_id = 0, mutate_prob = 1.0, grow_edge_prob = 1.0)
  discard generate_hyper_weights(hyperneat_top_id, 0, @[2, 3, 5])
  remove_topology(hyperneat_top_id)

  # regular neat

  let neat_top_id = add_topology(3, 4, @[16, 16])
  init_nn(neat_top_id)
  init_population(neat_top_id, 3)

  benchmark("population", 10):
    discard evaluate_population(neat_top_id, @[@[2.0, 3.0, 5.0], @[3.0, 5.0, 7.0], @[3.0, 5.0, 7.0]])

  remove_topology(neat_top_id)
