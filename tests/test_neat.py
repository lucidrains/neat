from neat.neat import (
    Topology
)

def test_add_neuron_and_synapse():
    top = Topology(2, 1)
    assert top.add_neuron() == 3
    assert top.add_neuron() == 4
    assert top.add_synapse( 0, 3) == 2

# mlp with population dim

import jax
import jax.numpy as jnp

from neat.neat import (
    init_mlp_weights_biases,
    mlp,
)

def test_population_mlp():
    weights, biases = init_mlp_weights_biases(10, 16, 16, 5, pop_dim = 8)

    action_logits = mlp(weights, biases, jnp.zeros((8, 10)))
    assert action_logits.shape == (8, 5)

def test_hyper():
    top = Topology(2, 1)
    top.init_population(10)
    weight = top.generate_hyper_weight(0, (3, 5))
    assert weight.shape == (3, 5)

    weights = top.generate_all_hyper_weights(10, (3, 5))
    assert len(weights) == 10
