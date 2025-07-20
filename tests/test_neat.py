# mlp with population dim

import jax.numpy as jnp

from neat.neat import (
    Topology,
    HyperNEAT,
    NEAT,
)

def test_hyperneat():
    pop = HyperNEAT(10, 16, 16, 5, num_hiddens = 16, pop_size = 8)

    action_logits = pop.forward(jnp.zeros((8, 10)))
    pop.genetic_algorithm_step(jnp.ones((8,)), num_selected = 4)

    assert action_logits.shape == (8, 5)

def test_neat():
    pop = NEAT(10, 16, 5, pop_size = 8)
    action_logits = pop.forward(jnp.zeros((8, 10)))
    pop.genetic_algorithm_step(jnp.ones((8,)), num_selected = 4)

    assert action_logits.shape == (8, 5)

def test_hyper_mlp():
    top = Topology(num_inputs = 2, num_outputs = 1, pop_size = 10, shape = (3, 5))
    weights = top.generate_hyper_weights()
    assert weights.shape == (10, 3, 5)
