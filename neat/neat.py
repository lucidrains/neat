from __future__ import annotations

import json
from random import randrange
from contextlib import contextmanager

import numpy as np
from jax import (
    numpy as jnp,
    random,
    jit,
    nn,
    Array,
)

from jax.tree_util import tree_map

from einops import einsum

import nimporter

from neat.neat_nim import (
    add_topology,
    remove_topology,
    init_population as init_population_nim,
    generate_hyper_weights as generate_hyper_weights_nim,
    generate_all_hyper_weights,
    crossover_and_add_to_population,
    select_and_tournament,
    add_node,
    add_edge,
    mutate_all,
    evaluate_nn_single,
    evaluate_population,
    get_topology_info,
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling

def log(t, eps = 1e-20):
    return jnp.log(t + eps)

def gumbel_noise(t):
    key = random.PRNGKey(randrange(int(1e6)))
    noise = random.uniform(key, t.shape)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature
        t = t + gumbel_noise(t)

    return t.argmax(axis = -1).tolist()

# topology

class Topology:
    def __init__(
        self,
        num_inputs,
        num_outputs,
        pop_size,
        num_hiddens = 32,
        shape: tuple[int, ...] | None = None,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None
    ):
        if isinstance(num_hiddens, int):
            num_hiddens = (num_hiddens,)

        self.id = add_topology(num_inputs, num_outputs, num_hiddens, mutation_hyper_params, crossover_hyper_params, selection_hyper_params)

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
    ) -> list[Array]:

        shape = default(shape, self.shape)

        all_weights = generate_all_hyper_weights(self.id, shape)
        all_weights = jnp.array(all_weights).reshape(-1, *shape)

        return all_weights

# mlp for actor

@jit
def mlp(
    weights: list[Array],
    biases: list[Array],
    t: Array
):
    weights_biases = list(zip(weights, biases))

    for weight, bias in weights_biases[:-1]:
        residual = t

        t = einsum(weight, t, '... i o, ... i -> ... o') + bias
        t = nn.relu(t)

        if residual.shape[-1] == t.shape[-1]:
            t = t + residual

    # last layer

    weight, bias = weights_biases[-1]
    return einsum(weight, t, '... i o, ... i -> ... o') + bias

class GeneticAlgorithm:
    def stats(self):
        return [get_topology_info(top_id) for top_id in self.all_top_ids]

    def genetic_algorithm_step(
        self,
        fitnesses: Array,
        selection_hyper_params = dict(),
        mutation_hyper_params = dict(),
        crossover_hyper_params = dict()
    ):

        # 1. selection
        # 2. tournament -> parent pairs

        (
            sel_indices,
            fitnesses,
            couples
        ) = select_and_tournament(self.all_top_ids, fitnesses.tolist(), selection_hyper_params)

        # 3. compute children with crossover
        # 4. concat children to population

        crossover_and_add_to_population(self.all_top_ids, couples, crossover_hyper_params)

        # 5. mutation

        mutate_all(self.all_top_ids, mutation_hyper_params)

class HyperNEAT(GeneticAlgorithm):
    def __init__(
        self,
        *dims,
        pop_size,
        num_hiddens = 0,
        weight_norm = True,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None
    ):

        self.dims = dims
        assert len(dims) > 1

        self.dim_pairs = list(zip(dims[:-1], dims[1:]))
        self.pop_size = pop_size

        hyper_weights_nn = []
        hyper_biases_nn = []

        evolution_hyper_params = dict(
            mutation_hyper_params = mutation_hyper_params,
            crossover_hyper_params = crossover_hyper_params,
            selection_hyper_params = selection_hyper_params
        )

        for dim_in, dim_out in self.dim_pairs:
            weight_shape = (dim_in, dim_out)
            bias_shape = (dim_out,)

            hyper_weights_nn.append(Topology(2, 1, pop_size, num_hiddens = num_hiddens, shape = weight_shape, **evolution_hyper_params))
            hyper_biases_nn.append(Topology(1, 1, pop_size, num_hiddens = num_hiddens, shape = bias_shape, **evolution_hyper_params))

        self.hyper_weights_nn = hyper_weights_nn
        self.hyper_biases_nn = hyper_biases_nn

        self.weight_norm = weight_norm

        self.all_top_ids = [top.id for top in (self.hyper_weights_nn + self.hyper_biases_nn)]

        self.generate_hyper_weights_and_biases()

    def generate_hyper_weights_and_biases(self):

        self.weights = [nn.generate_hyper_weights() for nn in self.hyper_weights_nn]
        self.biases = [nn.generate_hyper_weights() for nn in self.hyper_biases_nn]

        if self.weight_norm:
            # do a weight norm

            self.weights = [w / jnp.linalg.norm(w, axis = (-1, -2), keepdims = True) for w in self.weights]

    def genetic_algorithm_step(
        self,
        *args,
        **kwargs
    ):
        super().genetic_algorithm_step(*args, **kwargs)

        # regenerate hyperweights and biases

        self.generate_hyper_weights_and_biases()

    def single_forward(
        self,
        index: int,
        state: Array,
        sample = False,
        temperature = 1.
    ):
        single_weight, single_bias = tree_map(lambda t: t[index], (self.weights, self.biases))
        logits = mlp(single_weight, single_bias, state)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

    def forward(
        self,
        state: Array,
        sample = False,
        temperature = 1.
    ):
        logits = mlp(self.weights, self.biases, state)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

class NEAT(GeneticAlgorithm):
    def __init__(
        self,
        *dims,
        pop_size,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None
    ):
        self.dims = dims
        assert len(dims) >= 2

        self.pop_size = pop_size

        dim_in, *dim_hiddens, dim_out = dims

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output = np.empty((pop_size, self.dim_out), dtype = np.float32)

        self.top = Topology(dim_in, dim_out, num_hiddens = dim_hiddens, pop_size = pop_size, mutation_hyper_params = mutation_hyper_params, crossover_hyper_params = crossover_hyper_params, selection_hyper_params = selection_hyper_params)
        self.all_top_ids = [self.top.id]

    def single_forward(
        self,
        index: int,
        state: Array,
        sample = False,
        temperature = 1.
    ):
        logits = evaluate_nn_single(self.top.id, index, state.tolist(), use_exec_cache = True)

        logits = jnp.array(logits)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

    def forward(
        self,
        state: Array,
        sample = False,
        temperature = 1.,
    ):
        input = np.array(state, dtype = np.float32)

        evaluate_population(self.top.id, input, self.output)

        logits = jnp.array(self.output)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)
