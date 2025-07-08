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
    select_and_tournament,
    add_node,
    add_edge,
    crossover,
    mutate,
    evaluate_nn
)

from joblib import Parallel, delayed

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# topology

class Topology:
    def __init__(
        self,
        num_inputs,
        num_outputs,
        pop_size,
        shape: tuple[int, ...] | None = None
    ):
        self.id = add_topology(num_inputs, num_outputs)

        self.init_population(pop_size)

        self.pop_size = pop_size
        self.shape = shape

        if exists(shape):
            assert len(shape) == num_inputs

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
        shape: tuple[int, ...] | None = None
    ) -> Array:
        shape = default(shape, self.shape)

        seq_floats = generate_hyper_weights_nim(self.id, nn_id, shape)

        return jnp.array(seq_floats).reshape(shape)

    def generate_hyper_weights(
        self,
        shape: tuple[int, ...] | None = None,
        n_jobs = -1
    ) -> list[Array]:

        shape = default(shape, self.shape)

        all_weights = Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(self.generate_hyper_weight)(nn_id, shape) for nn_id in range(self.pop_size))
        return jnp.stack(all_weights)

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
        assert len(dims) > 1

        self.dim_pairs = list(zip(dims[:-1], dims[1:]))
        self.pop_size = pop_size

        hyper_weights_nn = []
        hyper_biases_nn = []

        for dim_in, dim_out in self.dim_pairs:
            weight_shape = (dim_in, dim_out)
            bias_shape = (dim_out,)

            hyper_weights_nn.append(Topology(2, 1, pop_size, shape = weight_shape))
            hyper_biases_nn.append(Topology(1, 1, pop_size, shape = bias_shape))

        self.hyper_weights_nn = hyper_weights_nn
        self.hyper_biases_nn = hyper_biases_nn

        self.generate_hyper_weights_and_biases()

    def generate_hyper_weights_and_biases(self):

        self.weights = [nn.generate_hyper_weights() for nn in self.hyper_weights_nn]
        self.biases = [nn.generate_hyper_weights() for nn in self.hyper_biases_nn]

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

        pass

    def forward(self, t: Array):
        return mlp(self.weights, self.biases, t)
