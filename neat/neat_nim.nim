import nimpy

import std/[
  random,
  assertions,
  math,
  sequtils,
  sets,
  setutils,
  tables,
  sugar,
  algorithm
]

import bitvector

import malebolgia

import arraymancer

type
  Prob = range[0.0..1.0]

# init

randomize()

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
  NodeType = enum
    input, output, hidden

  Node = object
    id: int
    `type`: NodeType = hidden

  Edge = object
    id: int
    from_node_id: int
    to_node_id: int

  MetaNode = ref object
    node_id: int
    disabled: bool
    can_disable: bool
    can_change_activation: bool = true
    activation: Activation = tanh
    bias: float = 0.0

  MetaEdge = ref object
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
    meta_nodes: seq[MetaNode] = @[]
    meta_edges: seq[MetaEdge] = @[]

  Topology = ref object
    id: int
    nodes: seq[Node] = @[] # nodes will be always arrange [input] [output] [hiddens]
    edges: seq[Edge] = @[]

    num_inputs: int
    num_outputs: int

    nn_id: int = 0
    node_innovation_id: int = 0
    edge_innovation_id: int = 0

    pop_size: int = 0
    population: seq[NeuralNetwork] = @[]

# globals

var topologies = init_table[int, Topology]()
var topology_id = 0

# functions

proc add_node(topology_id: int, node_type: NodeType = hidden): int
proc add_edge(topology_id: int, from_node_id: int, to_node_id: int): int

proc activate(act: Activation, input: Tensor[float]): Tensor[float]
proc activate(act: Activation, input: float): float
proc activate(node: MetaNode, input: Tensor[float]): Tensor[float]

proc add_topology(
  num_inputs: int,
  num_outputs: int,
  num_hiddens: seq[int]
): int {.exportpy.} =

  let topology = Topology(
    id: topology_id,
    num_inputs: num_inputs,
    num_outputs: num_outputs
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

proc add_node(
  topology_id: int,
  node_type: NodeType = hidden
): int {.exportpy.} =

  let top = topologies[topology_id]

  # create node, increment primary key, and add to global nodes

  let node = Node(id: top.node_innovation_id)
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
    from_node_id: from_node_id,
    to_node_id: to_node_id
  )

  top.edges.add(edge)

  top.edge_innovation_id += 1
  return edge.id

# main evolutionary functions

# population functions

proc init_nn(
  top_id: int
) =
  let top = topologies[top_id]

  let nn = NeuralNetwork()
  nn.num_inputs = top.num_inputs
  nn.num_outputs = top.num_outputs

  let
    output_node_index_start = top.num_inputs
    hidden_node_index_start = top.num_inputs + top.num_outputs
    hidden_edge_index_start = top.num_inputs * top.num_outputs

  # initialize input nodes

  for node in top.nodes[0..<top.num_inputs]:

    let meta_node = MetaNode(
      node_id: node.id,
      disabled: false,
      can_disable: false,
      can_change_activation: true,
      activation: tanh
    )

    nn.meta_nodes.add(meta_node)

  # initial output nodes

  
  for node in top.nodes[output_node_index_start..<(output_node_index_start + top.num_outputs)]:

    let meta_node = MetaNode(
      node_id: node.id,
      disabled: false,
      can_change_activation: false,
      activation: sigmoid
    )

    nn.meta_nodes.add(meta_node)

  # create edges - start off with only fully connected from inputs to outputs

  for edge in top.edges[0..<(top.num_inputs * top.num_outputs)]:

      let meta_edge = MetaEdge(
        edge_id: edge.id,
        disabled: false,
        weight: random_normal()
      )

      nn.meta_edges.add(meta_edge)

  # hiddens

  for node in top.nodes[hidden_node_index_start..<top.nodes.len]:

    let meta_node = MetaNode(
      node_id: node.id,
      disabled: coin_flip(),
      can_disable: true,
      bias: random_normal(),
      can_change_activation: true,
      activation: tanh
    )

    nn.meta_nodes.add(meta_node)

  for edge in top.edges[hidden_edge_index_start..<top.edges.len]:

    let meta_edge = MetaEdge(
      edge_id: edge.id,
      disabled: coin_flip(),
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

proc evaluate_nn_single(
  top_id: int,
  nn_id: int,
  inputs: seq[float]
): seq[float] {.exportpy.} =

  let seq_inputs = inputs.map(value => @[value])

  let seq_outputs = evaluate_nn(top_id, nn_id, seq_inputs)

  return seq_outputs.map(tensor => tensor[0])

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

  var coors = linspace(0.0, 1.0, shape[0])
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
): Tensor[float] =

  return activate(node.activation, input)

proc activate(
  act: Activation,
  input: Tensor[float]
): Tensor[float] =

  return input.map(value => activate(act, value))

proc activate(
  act: Activation,
  input: float
): float =

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

# mutation and crossover

proc tournament(
  fitnesses: seq[float],
  num_tournaments: range[1..int.high],
  tournament_size: range[2..int.high]
): seq[((int, float), (int, float))] {.exportpy.} =

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
  mutate_prob: Prob = 0.05,
  add_remove_edge_prob: Prob = 0.01,
  add_remove_node_prob: Prob = 0.01,
  change_activation_prob: Prob = 0.01,
  change_edge_weight_prob: Prob = 0.01,
  change_node_bias_prob: Prob = 0.01,
  decay_edge_weight_prob: Prob = 0.005,
  decay_node_bias_prob: Prob = 0.005,
  perturb_weight_strength: Prob = 0.1,
  perturb_bias_strength: Prob = 0.1,
  decay_factor: float = 0.95
) {.exportpy.} =

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  let activations = Activation.to_seq

  if not satisfy_prob(mutate_prob):
    return

  for meta_node in nn.meta_nodes:

    # add or removing an existing node, if valid (input and output nodes are preserved)

    if meta_node.can_disable and satisfy_prob(add_remove_node_prob):
      meta_node.disabled = meta_node.disabled xor true

    # mutating an activation on a node

    if meta_node.can_change_activation and satisfy_prob(change_activation_prob):
      let rand_activation_index = rand(Activation.high.ord)
      meta_node.activation = activations[rand_activation_index]

    if not meta_node.disabled and satisfy_prob(change_node_bias_prob):
      meta_node.bias += random_normal() * perturb_bias_strength

    if not meta_node.disabled and satisfy_prob(decay_node_bias_prob):
      meta_node.bias *= decay_factor

  for meta_edge in nn.meta_edges:

    # enabling / disabling an edge

    if satisfy_prob(add_remove_edge_prob):
      meta_edge.disabled = meta_edge.disabled xor true

    # changing a weight

    if not meta_edge.disabled and satisfy_prob(change_edge_weight_prob):
      meta_edge.weight += random_normal() * perturb_weight_strength

    if not meta_edge.disabled and satisfy_prob(decay_edge_weight_prob):
      meta_edge.weight *= decay_factor

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
    joint_node_ids &= (parent1_node_set -+- parent2_node_set).to_seq
    joint_edge_ids &= (parent1_edge_set -+- parent2_edge_set).to_seq

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

  # joint nodes / edges

  for node_id in joint_node_ids:

    let rand_node = if coin_flip():
      parent1_nodes_index[node_id]
    else:
      parent2_nodes_index[node_id]

    child_nodes.add(rand_node)

  for edge_id in joint_edge_ids:

    let rand_edge = if coin_flip():
      parent1_edges_index[edge_id]
    else:
      parent2_edges_index[edge_id]

    child_edges.add(rand_edge)

  # handle disjoint / excess genes

  for node_id in disjoint_node_ids:
    child_nodes.add(disjoint_nodes_index[node_id])

  for edge_id in disjoint_edge_ids:
    child_edges.add(disjoint_edges_index[edge_id])

  # add child to population

  let nn = NeuralNetwork(
    num_inputs: top.num_inputs,
    num_outputs: top.num_outputs,
    meta_nodes: child_nodes,
    meta_edges: child_edges
  )

  return nn

proc crossover_and_add_to_population(
  top_id: int,
  parent_indices_and_fitnesses: seq[((int, float), (int, float))],
  fitness_diff_is_same: float = 0.0
) {.exportpy.} =

  let top = topologies[top_id]

  for one_pair in parent_indices_and_fitnesses:
    let (parent1_info, parent2_info) = one_pair

    let (parent1, fitness1) = parent1_info
    let (parent2, fitness2) = parent2_info

    let child = crossover(top_id, parent1, parent2, fitness1, fitness2, fitness_diff_is_same)

    top.population.add(child)

# quick test

when is_main_module:
  let top_id = add_topology(3, 1, @[16, 16])
  init_nn(top_id)
  init_population(top_id, 10)
  discard generate_hyper_weights(top_id, 0, @[2, 3, 5])
  remove_topology(top_id)