import std/[
  random,
  assertions,
  tables,
  options
]

import ../neat/neat_nim

proc test_tournament() =
  # standard tournament
  let fitnesses = @[1.0'f32, 5.0'f32, 3.0'f32, 8.0'f32, 2.0'f32]
  let result = tournament(fitnesses, num_tournaments=1, tournament_size=3)

  assert result.len == 1
  let couple = result[0]
  assert couple.parent1.fitness >= couple.parent2.fitness
  assert couple.parent1.index >= 0 and couple.parent1.index < fitnesses.len
  assert couple.parent2.index >= 0 and couple.parent2.index < fitnesses.len
  assert couple.parent1.index != couple.parent2.index

  # tournament with all low(float32) or tied fitnesses
  let tied_fits = @[low(float32), low(float32), low(float32)]
  let tied_result = tournament(tied_fits, num_tournaments=2, tournament_size=2)
  assert tied_result.len == 2
  for c in tied_result:
    assert c.parent1.index >= 0 and c.parent1.index < tied_fits.len
    assert c.parent2.index >= 0 and c.parent2.index < tied_fits.len

proc test_fuss_couples() =
  # diverse fitnesses
  let fits = @[1.0'f32, 2.0'f32, 3.0'f32, 10.0'f32]
  let couples = fuss_couples(fits, 10)
  assert couples.len == 10
  for c in couples:
    assert c.parent1.index >= 0 and c.parent1.index < fits.len
    assert c.parent2.index >= 0 and c.parent2.index < fits.len

  # identical fitnesses
  let same_fits = @[5.0'f32, 5.0'f32, 5.0'f32]
  let same_couples = fuss_couples(same_fits, 5)
  assert same_couples.len == 5
  for c in same_couples:
    assert c.parent1.index >= 0 and c.parent1.index < same_fits.len
    assert c.parent2.index >= 0 and c.parent2.index < same_fits.len

proc test_sample_waiting_time() =
  # boundary probabilities
  assert sample_waiting_time(0.0) == int.high
  assert sample_waiting_time(-0.1) == int.high
  assert sample_waiting_time(1.0) == 1
  assert sample_waiting_time(1.5) == 1

  # random trials should always be >= 1 and positive
  for _ in 0 ..< 1000:
    let w = sample_waiting_time(0.5)
    assert w >= 1

proc test_topology_construction() =
  # no hiddens: 2 inputs, 1 output -> 3 nodes, 2 direct edges
  let top_empty_id = add_topology(2, 1, @[])
  let info_empty = get_topology_info(top_empty_id)
  assert info_empty.total_innovated_nodes == 3
  assert info_empty.total_innovated_edges == 2

  # multi-layer hidden: 2 inputs, 1 output, [3, 3] hiddens
  # nodes: 2 + 1 + 3 + 3 = 9 nodes
  # edges: 2 direct (in->out) + (2*3 in->h1) + (3*3 h1->h2) + (3*1 h2->out) = 2 + 6 + 9 + 3 = 20 edges
  let top_multi_id = add_topology(2, 1, @[3, 3])
  let info_multi = get_topology_info(top_multi_id)
  assert info_multi.total_innovated_nodes == 9
  assert info_multi.total_innovated_edges == 20

proc test_activation_and_aggregation() =
  let top_id = add_topology(2, 1, @[1])
  init_population(top_id, 2)
  let top = topologies[top_id]
  let nn = top.population[0]

  # input and output nodes cannot change activation or aggregation

  for i in 0 .. 2:
    assert not nn.meta_nodes[i].can_change_activation
    assert not nn.meta_nodes[i].can_change_aggregation

  let hidden_node = nn.meta_nodes[3]
  assert hidden_node.can_change_activation
  assert hidden_node.can_change_aggregation
  assert hidden_node.aggregation == Aggregation.sum

  # inputs [2, 4] -> hidden weighted sum 4 + 12, bias 0.5

  nn.meta_edges[0].disabled = true
  nn.meta_edges[1].disabled = true
  nn.meta_edges[2].disabled = false
  nn.meta_edges[2].weight = 2.0'f32
  nn.meta_edges[3].disabled = false
  nn.meta_edges[3].weight = 3.0'f32
  nn.meta_edges[4].disabled = false
  nn.meta_edges[4].weight = 1.0'f32

  hidden_node.bias = 0.5'f32
  hidden_node.activation = identity
  hidden_node.disabled = false
  nn.meta_nodes[2].bias = 0.0'f32
  nn.meta_nodes[2].activation = identity
  nn.meta_nodes[2].disabled = false

  let test_inputs = @[2.0'f32, 4.0'f32]
  let cases = [
    (Aggregation.sum, 16.5'f32),
    (Aggregation.product, 48.5'f32),
    (Aggregation.max, 12.5'f32),
    (Aggregation.min, 4.5'f32),
    (Aggregation.mean, 8.5'f32)
  ]

  for (aggregation, expected) in cases:
    hidden_node.aggregation = aggregation
    nn.cached_exec_trace = none(ExecTrace)

    let out_uncached = evaluate_nn_single(top_id, 0, test_inputs, use_exec_cache = false)
    let out_cached = evaluate_nn_single(top_id, 0, test_inputs, use_exec_cache = true)

    assert abs(out_uncached[0] - expected) < 1e-5'f32
    assert abs(out_cached[0] - expected) < 1e-5'f32

  var custom_hparams = MutationHyperParams(
    mutate_prob: 1.0,
    change_activation_prob: 1.0,
    change_aggregation_prob: 1.0,
    add_novel_edge_prob: 0.0,
    grow_edge_prob: 0.0,
    grow_node_prob: 0.0,
    toggle_meta_edge_prob: 0.0,
    add_remove_node_prob: 0.0
  )
  discard mutate_structure(top, 1, some(custom_hparams), force = true)
  let mutated_hidden = top.population[1].meta_nodes[3]
  assert mutated_hidden.can_change_activation
  assert mutated_hidden.can_change_aggregation

when is_main_module:
  randomize(42)
  test_tournament()
  test_fuss_couples()
  test_sample_waiting_time()
  test_topology_construction()
  test_activation_and_aggregation()
  echo "ALL NIM UNIT TESTS PASSED!"
