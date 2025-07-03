import nimpy

import std/[
  random,
  assertions,
  math,
  sequtils,
  sugar,
  algorithm
]

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
    activation: Activation = tanh
    bias: float = 0.0

  MetaEdge = ref object
    edge_id: int
    disabled: bool
    weight: float

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

var topologies: seq[Topology] = @[]
var topology_id = 0

# functions

proc add_node(topology_id: Natural, node_type: NodeType = hidden): Natural
proc add_edge(topology_id: Natural, from_node_id: Natural, to_node_id: Natural): Natural

proc add_topology(
  num_inputs: int,
  num_outputs: int
): int {.exportpy.} =

  let topology = Topology(
    id: topology_id,
    num_inputs: num_inputs,
    num_outputs: num_outputs
  )

  topologies.add(topology)

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

  # return id

  return topology.id

proc add_node(
  topology_id: Natural,
  node_type: NodeType = hidden
): Natural {.exportpy.} =

  let top = topologies[topology_id]

  # create node, increment primary key, and add to global nodes

  let node = Node(id: top.node_innovation_id)
  top.nodes.add(node)

  top.node_innovation_id += 1
  return node.id

proc add_edge(
  topology_id: Natural,
  from_node_id: Natural,
  to_node_id: Natural
): Natural {.exportpy.} =

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
  top_id: Natural,
) =
  let top = topologies[top_id]

  let nn = NeuralNetwork()
  nn.num_inputs = top.num_inputs
  nn.num_outputs = top.num_outputs

  # initialize nodes

  for node_id in 0..<top.num_inputs:
    let node = top.nodes[node_id]

    let meta_node = MetaNode(
      node_id: node.id,
      disabled: false,
      can_disable: false,
      activation: tanh
    )

    nn.meta_nodes.add(meta_node)

  for i in 0..<top.num_outputs:
    let node_id = top.num_inputs + i
    let node = top.nodes[node_id]

    let meta_node = MetaNode(
      node_id: node.id,
      disabled: false,
      activation: sigmoid
    )

    nn.meta_nodes.add(meta_node)

  # create edges - start off with only fully connected from inputs to outputs

  for edge_id in 0..<(top.num_inputs * top.num_outputs):
      let edge = top.edges[edge_id]

      let meta_edge = MetaEdge(
        edge_id: edge.id,
        disabled: false,
        weight: random_normal()
      )

      nn.meta_edges.add(meta_edge)

  # add neural net to population

  top.population.add(nn)

proc init_population(
  top_id: Natural,
  pop_size: range[1..int.high],
) =

  let top = topologies[top_id]
  assert top.pop_size == 0

  top.pop_size = pop_size

  for _ in 0..<pop_size:
    init_nn(top_id)

# forward

proc evaluate_nn(
  top_id: Natural,
  nn_id: Natural,
  inputs: seq[float]
): seq[float] {.exportpy.} =

  discard

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
  top_id: Natural,
  fitnesses: seq[float],
  num_tournaments: int,
  tournament_size: int
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

proc select(
  top_id: Natural,
  fitnesses: seq[float],
  num_selected: range[1..int.high]
): seq[int] {.exportpy.} =

  let sorted_indices = fitnesses
    .to_tensor()
    .argsort(order = SortOrder.Descending)
    .to_flat_seq()

  return sorted_indices[0..<num_selected]

proc mutate(
  top_id: Natural,
  nn_id: Natural,
  add_remove_edge_prob: Prob = 0.0,
  add_remove_node_prob: Prob = 0.0,
  change_activation_prob: Prob = 0.0,
  change_edge_weight_prob: Prob = 0.0,
  change_node_bias_prob: Prob = 0.0,
  perturb_weight_strength: Prob = 0.1,
  perturb_bias_strength: Prob = 0.1
) {.exportpy.} =

  let top = topologies[top_id]
  let nn = top.population[nn_id]

  let activations = Activation.to_seq

  for meta_node in nn.meta_nodes:

    # add or removing an existing node, if valid (input and output nodes are preserved)

    if meta_node.can_disable and satisfy_prob(add_remove_node_prob):
      meta_node.disabled = meta_node.disabled xor true

    # mutating an activation on a node

    if satisfy_prob(change_activation_prob):
      let rand_activation_index = rand(Activation.high.ord)
      meta_node.activation = activations[rand_activation_index]

    if not meta_node.disabled and satisfy_prob(change_node_bias_prob):
      meta_node.bias += random_normal() * perturb_bias_strength

  for meta_edge in nn.meta_edges:

    # enabling / disabling an edge

    if satisfy_prob(add_remove_edge_prob):
      meta_edge.disabled = meta_edge.disabled xor true

    # changing a weight

    if not meta_edge.disabled and satisfy_prob(change_edge_weight_prob):
      meta_edge.weight += random_normal() * perturb_weight_strength

proc crossover(
  top_id: Natural,
  first_parent_nn_id: Natural,
  second_parent_nn_id: Natural,
  first_parent_fitness: float,
  second_parent_fitness: float
): NeuralNetwork {.exportpy.} =
  discard

# quick test

when is_main_module:
  let top_id = add_topology(2, 1)

  init_population(top_id, pop_size = 8)
  mutate(top_id, 0, change_activation_prob = 1.0)
