from __future__ import annotations
from random import randrange

import jax
from jax import (
    numpy as jnp,
    vmap,
    random,
    jit,
    nn,
    Array,
)

from jax.tree_util import tree_map

from itertools import product

import einx
from einops import einsum

import nimporter

from neat.neat_nim import (
    add_topology,
    remove_topology,
    init_population as init_population_nim,
    init_top_lock,
    deinit_top_lock,
    generate_hyper_weights as generate_hyper_weights_nim,
    crossover_and_add_to_population,
    select_and_tournament,
    add_node,
    add_edge,
    mutate,
    evaluate_nn,
    evaluate_nn_single,
    evaluate_population
)

from joblib import Parallel, delayed

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
        shape: tuple[int, ...] | None = None
    ):
        if isinstance(num_hiddens, int):
            num_hiddens = (num_hiddens,)

        self.id = add_topology(num_inputs, num_outputs, num_hiddens)

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
        residual = t

        t = einsum(weight, t, '... i o, ... i -> ... o') + bias
        t = nn.relu(t)

        if residual.shape[-1] == t.shape[-1]:
            t = t + residual

    # last layer

    weight, bias = weights_biases[-1]
    return einsum(weight, t, '... i o, ... i -> ... o') + bias

class GeneticAlgorithm:
    def genetic_algorithm_step(
        self,
        fitnesses: Array,
        num_selected = None,
        num_selected_frac = None,
        tournament_frac = 0.25,
        num_preserve_elites_frac = 0.1,
        n_jobs = -1
    ):
        assert exists(num_selected) ^ exists(num_selected_frac)

        if exists(num_selected_frac):
            assert 0. < num_selected_frac < 1.
            num_selected = min(2, int(self.pop_size * num_selected_frac))

        assert num_selected >= 2

        assert 0. <= num_preserve_elites_frac <= 1.
        assert 0. <= tournament_frac <= 1.

        num_preserve_elites = int(num_preserve_elites_frac * num_selected)
        assert num_preserve_elites < num_selected

        print(f'fitness: max {fitnesses.max():.2f} | mean {fitnesses.mean():.2f} | std {fitnesses.std():.2f}')

        tournament_size = max(2, int(tournament_frac * num_selected))

        # 1. selection
        # 2. tournament -> parent pairs

        (
            sel_indices,
            fitnesses,
            parent_indices_and_fitnesses
        )= select_and_tournament(self.all_top_ids, fitnesses.tolist(), num_selected, tournament_size)

        # 3. compute children with crossover
        # 4. concat children to population

        Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(crossover_and_add_to_population)(top_id, parent_indices_and_fitnesses) for top_id in self.all_top_ids)

        # 5. mutation

        for top_id in self.all_top_ids:
            init_top_lock(top_id)

        Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(mutate)(top_id, nn_id) for top_id, nn_id in product(self.all_top_ids, range(num_preserve_elites, self.pop_size)))

        for top_id in self.all_top_ids:
            deinit_top_lock(top_id)

class HyperNEAT(GeneticAlgorithm):
    def __init__(
        self,
        *dims,
        pop_size,
        num_hiddens = 0,
        weight_norm = True
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

            hyper_weights_nn.append(Topology(2, 1, pop_size, num_hiddens = num_hiddens, shape = weight_shape))
            hyper_biases_nn.append(Topology(1, 1, pop_size, num_hiddens = num_hiddens, shape = bias_shape))

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
    ):
        self.dims = dims
        assert len(dims) >= 2

        self.pop_size = pop_size

        dim_in, *dim_hiddens, dim_out = dims

        self.top = Topology(dim_in, dim_out, num_hiddens = dim_hiddens, pop_size = pop_size)
        self.all_top_ids = [self.top.id]

    
    def single_forward(
        self,
        index: int,
        state: Array,
        sample = False,
        temperature = 1.
    ):
        logits = evaluate_nn_single(self.top.id, index, state.tolist())

        logits = jnp.array(logits)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

    def forward(
        self,
        state: Array,
        sample = False,
        temperature = 1.,
        n_jobs = -1
    ):
        logits = Parallel(n_jobs = n_jobs, backend = 'threading')(delayed(evaluate_nn_single)(self.top.id, nn_id, one_state.tolist()) for nn_id, one_state in zip(range(self.pop_size), state))
        logits = jnp.array(logits)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)
