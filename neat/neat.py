from __future__ import annotations

import jax
from jax import (
    numpy as jnp,
    vmap,
    random,
    jit,
    nn,
    Array,
)

import einx
from einops import einsum

import nimporter

from neat.neat_nim import (
    add_topology,
    remove_topology,
    init_population as init_population_nim,
    generate_hyper_weights as generate_hyper_weights_nim,
    add_node,
    add_edge,
    tournament,
    crossover,
    select,
    mutate,
    evaluate_nn
)

from joblib import Parallel, delayed

# functions

class Topology:
    def __init__(
        self,
        num_inputs,
        num_outputs
    ):
        self.id = add_topology(num_inputs, num_outputs)

    def __del__(self):
        remove_topology(self.id)

    def init_population(self, pop_size):
        return init_population_nim(self.id, pop_size)

    def add_neuron(self):
        return add_node(self.id)

    def add_synapse(self, from_id, to_id):
        return add_edge(self.id, from_id, to_id)

    def generate_hyper_weight(
        self,
        nn_id,
        shape: tuple[int, ...]
    ) -> Array:

        seq_floats = generate_hyper_weights_nim(self.id, nn_id, shape)

        return jnp.array(seq_floats).reshape(shape)

    def generate_all_hyper_weights(
        self,
        pop_size,
        shape: tuple[int, ...],
        n_jobs = -1
    ) -> list[Array]:

        return Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(self.generate_hyper_weight)(nn_id, shape) for nn_id in range(pop_size))

# mlp for actor

@jit
def mlp(
    weights: list[Array],
    biases: list[Array],
    t: Array
):
    weights_biases = list(zip(weights, biases))

    for weight, bias in weights_biases[:-1]:
        t = einsum(weight, t, '... i o, ... i -> ... o') + bias
        t = nn.silu(t)

    # last layer

    weight, bias = weights_biases[-1]
    return einsum(weight, t, '... i o, ... i -> ... o') + bias

class PopulationMLP:
    def __init__(
        self,
        *dims,
        pop_size
    ):
        self.dims = dims
        self.pop_size = pop_size
        self.init_mlp_weights_biases()

    def init_mlp_weights_biases(
        self,
        add_pop_dim = True,
        seed = 42
    ):
        dims = self.dims
        pop_dim = self.pop_size
        assert len(dims) > 1

        weights = []
        biases = []

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):

            weight_shape = (dim_in, dim_out)
            bias_shape = (dim_out,)

            if add_pop_dim:
                weight_shape = (pop_dim, *weight_shape)
                bias_shape = (pop_dim, *bias_shape)

            weights.append(jnp.zeros(weight_shape))
            biases.append(jnp.zeros(bias_shape))

        self.weights = weights
        self.biases = biases

        return weights, biases

    def genetic_algorithm_step(
        self,
        fitnesses: Array,
        top_id: int = 0,
        num_selected = 2
    ):
        assert num_selected >= 2

        policy_weights, policy_biases = (self.weights, self.biases)

        # todo
        # 1. selection
        # 2. tournament -> parent pairs
        # 3. compute children with crossover
        # 4. concat children to population
        # 5. mutation

        selected_indices = select(top_id, fitnesses.tolist(), num_selected)

        return policy_weights, policy_biases

    def forward(self, t: Array):
        return mlp(self.weights, self.biases, t)
