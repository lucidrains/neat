from neat.neat import (
    add_neat_topology,
    add_neuron,
    add_synapse
)

def test_add_neuron_and_synapse():
    top_id = add_neat_topology(2, 1)
    assert add_neuron(top_id) == 3
    assert add_neuron(top_id) == 4
    assert add_synapse(top_id, 0, 3) == 2

# mlp with population dim

import jax
import jax.numpy as jnp

from neat.neat import (
    init_mlp_weights_biases,
    mlp,
    generate_hyper_weight,
    generate_all_hyper_weights,
    add_topology,
    init_population
)

def test_population_mlp():
    weights, biases = init_mlp_weights_biases(10, 16, 16, 5, pop_dim = 8)

    action_logits = mlp(weights, biases, jnp.zeros((8, 10)))
    assert action_logits.shape == (8, 5)

def test_hyper():
    top_id = add_topology(2, 1)
    init_population(top_id, 10)
    weight = generate_hyper_weight(top_id, 0, (3, 5))
    assert weight.shape == (3, 5)

    weights = generate_all_hyper_weights(top_id, 10, (3, 5))
    assert len(weights) == 10
