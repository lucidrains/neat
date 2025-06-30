from neat.neat import (
    add_neat_topology,
    add_neuron,
    add_synapse
)

def test_add_neuron_and_synapse():
    top_id = add_neat_topology(2, 1)
    assert add_neuron(top_id) == 0
    assert add_neuron(top_id) == 1
    assert add_synapse(top_id, 0, 1) == 0

# mlp with population dim

import jax
import jax.numpy as jnp

from neat.neat import (
    init_mlp_weights_biases,
    mlp
)

def test_population_mlp():
    weights, biases = init_mlp_weights_biases(10, 16, 16, 5, pop_dim = 8)

    action_logits = mlp(weights, biases, jnp.zeros((8, 10)))
    assert action_logits.shape == (8, 5)
