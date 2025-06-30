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

# init

randomize()

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

  MetaNode = object
    node_id: int
    disabled: bool
    activation: Activation = tanh
    bias: float = 0.0

  MetaEdge = object
    edge_id: int
    disabled: bool
    weight: float

  NeuralNetwork = ref object
    id: int
    topology_id: int
    num_inputs: int
    num_outputs: int
    meta_nodes: seq[MetaNode] = @[] # nodes will be always arrange [input] [output] [hiddens]
    meta_edges: seq[MetaEdge] = @[]

  Topology = ref object
    id: int
    nodes: seq[Node] = @[]
    edges: seq[Edge] = @[]

    nn_id: int = 0
    node_innovation_id: int = 0
    edge_innovation_id: int = 0

    population: seq[NeuralNetwork] = @[]

# globals

var topologies: seq[Topology] = @[]
var topology_id = 0

# functions

proc add_topology(): int {.exportpy.} =
  let topology = Topology(id: topology_id)

  topologies.add(topology)

  topology_id += 1
  return topology.id

proc add_node(topology_id: int): int {.exportpy.} =

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

proc init_population(
  top_id: int,
  pop_size: int
) =
  discard

proc init_nn(
  top_id: int,
  num_inputs: int,
  num_outputs: int
) =
  discard

# forward

proc evaluate_nn(
  top_id: int,
  nn_id: int,
  inputs: seq[float]
): seq[float] {.exportpy.} =

  discard

proc activate(act: Activation, input: float): float =
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
  top_id: int,
  fitnesses: seq[float],
  num_tournaments: int,
  tournament_size: int
): seq[(int, int)] {.exportpy.} =

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

    result.add((parent1, parent2))

proc select(
  top_id: int,
  fitnesses: seq[float],
  num_selected: int
): seq[int] {.exportpy.} =

  let sorted_indices = fitnesses
    .to_tensor()
    .argsort(order = SortOrder.Descending)
    .to_flat_seq()

  return sorted_indices[0..<num_selected]

proc mutate(
  top_id: int,
  nn_id: int,
  add_remove_edge_prob: float = 0.0,
  add_remove_node_prob: float = 0.0,
  change_activation_prob: float = 0.0,
  change_edge_weight_prob: float = 0.0,
  change_node_bias_prob: float = 0.0
) {.exportpy.} =
  discard

proc crossover(
  top_id: int,
  first_parent_nn_id: int,
  second_parent_nn_id: int
) {.exportpy.} =
  discard

# quick test

when is_main_module:
  echo tournament(0, @[1.0, 3.0, 2.0, 4.0], 2, 3)
