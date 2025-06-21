import nimpy

import std/[
  random,
  assertions,
  math,
]

# init

randomize()

# types

type
  Node = object
    id: int

  Edge = object
    id: int
    from_node_id: int
    to_node_id: int

  MetaNode = object
    node_id: int
    disabled: bool

  MetaEdge = object
    edge_id: int
    disabled: bool
    weight: float

  NeuralNetwork = ref object
    id: int
    topology_id: int
    meta_nodes: seq[MetaNode] = @[]
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

# activations

proc sigmoid(x: float): float =
  return (1.0 / (1.0 + exp(-x)))

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

# mutation and crossover

proc mutate(top_id: int, nn_id: int) =
  assert false


proc crossover(
  top_id: int,
  first_parent_nn_id: int,
  second_parent_nn_id: int
) =
  assert false
