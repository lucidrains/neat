import nimpy

import std/[
  assertions
]

# types

type
  Node = object
    id: int

  Edge = object
    id: int
    from_node_id: int
    to_node_id: int

# globals

var node_innovation_id = 0
var edge_innovation_id = 0

var nodes: seq[Node] = @[]
var edges: seq[Edge] = @[]

# functions

proc add_node(): int {.exportpy.} =

  # create node, increment primary key, and add to global nodes

  let node = Node(id: node_innovation_id)
  nodes.add(node)

  node_innovation_id += 1
  return node.id

proc add_edge(
  from_node_id: int,
  to_node_id: int
): int {.exportpy.} =

  # validate node id

  let max_node_id = nodes.len
  assert(0 <= from_node_id  and from_node_id < max_node_id)
  assert(0 <= to_node_id and to_node_id < max_node_id)

  # create edge, increment primary key and add to global edges

  let edge = Edge(
    id: edge_innovation_id,
    from_node_id: from_node_id,
    to_node_id: to_node_id
  )

  edges.add(edge)

  edge_innovation_id += 1
  return edge.id
