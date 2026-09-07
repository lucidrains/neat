import numpy as np

from neat.neat import NEAT

def test_neat():
    pop = NEAT(10, 16, 5, pop_size = 8, mutation_hyper_params = dict(mutate_prob = 0.25))
    action_logits = pop.forward(np.zeros((8, 10)))
    pop.genetic_algorithm_step(np.ones((8,)))

    assert action_logits.shape == (8, 5)

def test_clune_connection_costs_and_modularity():
    pop = NEAT(4, 8, 2, pop_size = 10)
    costs = pop.connection_costs(squared = True)
    assert len(costs) == 1 and len(costs[0]) == 10
    assert all(c >= 0.0 for c in costs[0])

    mods = pop.modularities()
    assert len(mods) == 1 and len(mods[0]) == 10
    assert all(-1.0 <= m <= 1.0 for m in mods[0])

    # test genetic algorithm step with Clune connection cost penalty
    fits = np.random.randn(10).astype(np.float32)
    pop.genetic_algorithm_step(
        fits,
        connection_cost_penalty = 0.1
    )

    # test instance-level configuration
    pop_configured = NEAT(4, 1, pop_size = 10, connection_cost_penalty = 0.05)
    assert pop_configured.connection_cost_penalty == 0.05
    pop_configured.genetic_algorithm_step(fits)

