# mlp with population dim

import jax.numpy as jnp

from neat.neat import (
    Topology,
    NEAT,
)

def test_neat():
    pop = NEAT(10, 16, 5, pop_size = 8, mutation_hyper_params = dict(mutation_rate = 0.25))
    action_logits = pop.forward(jnp.zeros((8, 10)))
    pop.genetic_algorithm_step(jnp.ones((8,)))

    assert action_logits.shape == (8, 5)
