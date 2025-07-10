from neat.neat import (
    Topology
)

def test_add_neuron_and_synapse():
    top = Topology(2, 1, 1, num_hiddens = 0)
    assert top.add_neuron() == 3
    assert top.add_neuron() == 4
    assert top.add_synapse(0, 3) == 2

# mlp with population dim

import jax
import jax.numpy as jnp

from neat.neat import (
    PopulationMLP,
    mlp,
)

def test_population_mlp():
    pop = PopulationMLP(10, 16, 16, 5, num_hiddens = 16, pop_size = 8)

    action_logits = pop.forward(jnp.zeros((8, 10)))
    pop.genetic_algorithm_step(jnp.ones((8,)), num_selected = 4)

    assert action_logits.shape == (8, 5)

def test_hyper():
    top = Topology(num_inputs = 2, num_outputs = 1, pop_size = 10, shape = (3, 5))
    weights = top.generate_hyper_weights()
    assert weights.shape == (10, 3, 5)
