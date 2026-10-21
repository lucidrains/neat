import numpy as np
import pytest

from neat.neat import NEAT

def test_neat():
    pop = NEAT(10, 16, 5, pop_size = 8, mutation_hyper_params = dict(mutate_prob = 0.25))
    action_logits = pop.forward(np.zeros((8, 10)))
    pop.genetic_algorithm_step(np.ones((8,)))

    assert action_logits.shape == (8, 5)

def test_clune_connection_costs_and_modularity():
    pop = NEAT(4, 8, 2, pop_size = 10)
    costs = pop.connection_costs(squared = True)
    mods = pop.modularities()

    assert len(costs) == 1 and len(costs[0]) == 10
    assert all(c >= 0.0 for c in costs[0])
    assert len(mods) == 1 and len(mods[0]) == 10
    assert all(-1.0 <= m <= 1.0 for m in mods[0])

    fits = np.random.randn(10).astype(np.float32)
    pop.genetic_algorithm_step(fits, connection_cost_penalty = 0.1)

    pop = NEAT(4, 1, pop_size = 10, connection_cost_penalty = 0.05)
    assert pop.connection_cost_penalty == 0.05
    pop.genetic_algorithm_step(fits)

def test_aggregation_and_activation_mutation():
    mutation_hyper_params = dict(
        mutate_prob = 1.0,
        change_activation_prob = 0.5,
        change_aggregation_prob = 0.5,
    )
    pop = NEAT(4, 8, 2, pop_size = 10, mutation_hyper_params = mutation_hyper_params)
    inputs = np.random.randn(10, 4).astype(np.float32)

    pop.genetic_algorithm_step(np.random.rand(10).astype(np.float32))
    out = pop.forward(inputs)

    assert out.shape == (10, 2)
    assert not np.isnan(out).any()

def test_champion_index_preserved_post_selection():
    pop = NEAT(4, 6, 2, pop_size = 8, num_islands = 2, mutation_hyper_params = dict(mutate_prob = 0.0))
    x = np.random.randn(4).astype(np.float32)
    before = np.stack([pop.single_forward(idx, x) for idx in range(pop.top.pop_size)])

    pop.genetic_algorithm_step(np.array([1., 2., 50., 3., 4., 100., 5., 6.], dtype = np.float32))

    assert pop.champion_index == 4
    np.testing.assert_allclose(pop.champion(x), before[5], rtol = 1e-6, atol = 1e-6)

def test_unfreeze_preserves_input_output_flags_and_no_duplicate_genes():
    import json

    mutation_hyper_params = dict(
        mutate_prob = 1.0,
        grow_edge_prob = 0.8,
        grow_node_prob = 0.5,
        add_novel_edge_prob = 0.5,
        add_remove_node_prob = 0.8,
        change_activation_prob = 0.5,
        change_aggregation_prob = 0.5,
    )
    crossover_hyper_params = dict(prob_remove_disabled_node = 1.0)
    pop = NEAT(3, 2, pop_size = 8, mutation_hyper_params = mutation_hyper_params, crossover_hyper_params = crossover_hyper_params)

    for _ in range(5):
        pop.freeze()
        pop.unfreeze()
        pop.genetic_algorithm_step(np.random.randn(8).astype(np.float32))

        for genes in json.loads(pop.to_json()[0])[2]:
            node_ids = [n['node_id'] for n in genes['meta_nodes']]
            edge_ids = [e['edge_id'] for e in genes['meta_edges']]

            assert len(node_ids) == len(set(node_ids))
            assert len(edge_ids) == len(set(edge_ids))
            assert set(range(5)) <= set(node_ids)
            assert all(not n['disabled'] and not n['can_disable'] for n in genes['meta_nodes'][:5])

def test_backprop_respects_frozen():
    pop = NEAT(3, 4, 2, pop_size = 4)
    x = np.random.randn(3).astype(np.float32)
    y = np.random.randn(2).astype(np.float32)

    pop.freeze()
    out_before = pop.single_forward(0, x).copy()
    pop.backprop(0, x, y, learning_rate = 0.1)
    np.testing.assert_allclose(pop.single_forward(0, x), out_before, rtol = 1e-6, atol = 1e-6)

    pop.unfreeze()
    pop.backprop(0, x, y, learning_rate = 0.1)
    assert not np.allclose(pop.single_forward(0, x), out_before)

@pytest.mark.parametrize('time_mode', ('raw', 'sin_cos'))
def test_time_inputs(time_mode):
    pop = NEAT(4, 2, pop_size = 2, time_mode = time_mode)
    assert pop.forward(np.zeros((2, 4), dtype = np.float32), time = 0.5).shape == (2, 2)
