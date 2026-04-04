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

import nimporter

from neat.neat_nim import (
    add_topology,
    remove_topology,
    init_population as init_population_nim,
    crossover_and_add_to_population,
    select_and_tournament,
    add_node,
    add_edge,
    mutate_all,
    migrate_islands as migrate_nim,
    reset_top_islands as reset_islands_nim,
    evaluate_nn_single,
    evaluate_population,
    get_topology_info,
    save_json_to_file
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
        selection_hyper_params = None,
        num_islands = 1
    ):
        if isinstance(num_hiddens, int):
            num_hiddens = (num_hiddens,)

        self.id = add_topology(num_inputs, num_outputs, num_hiddens, mutation_hyper_params, crossover_hyper_params, selection_hyper_params, num_islands)

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

class GeneticAlgorithm:
    def stats(self):
        return [get_topology_info(top_id) for top_id in self.all_top_ids]

    def save_json(self, filename):
        for top_id in self.all_top_ids:
            save_json_to_file(top_id, f'{filename}.id.{top_id}.json')

    def genetic_algorithm_step(
        self,
        fitnesses: Array,
        selection_hyper_params = None,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        migrate_num = 0,
        reset_islands_num = 0,
        reset_islands_tournament_size = 3
    ):

        # 1. selection
        # 2. tournament -> parent pairs

        (
            sel_indices,
            sel_fitnesses,
            couples,
            target_nn_ids
        ) = select_and_tournament(self.all_top_ids, fitnesses.tolist(), selection_hyper_params)

        # 3. compute children with crossover
        # 4. concat children to population

        crossover_and_add_to_population(self.all_top_ids, couples, target_nn_ids, crossover_hyper_params)

        # 5. migration

        if migrate_num > 0:
            migrate_nim(self.all_top_ids, migrate_num)

        # 6. island reset

        if reset_islands_num > 0:
            reset_islands_nim(self.all_top_ids, fitnesses.tolist(), reset_islands_num, reset_islands_tournament_size)

        # 7. mutation

        mutate_all(self.all_top_ids, mutation_hyper_params)

class NEAT(GeneticAlgorithm):
    def __init__(
        self,
        *dims,
        pop_size,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None,
        num_islands = 1
    ):
        self.dims = dims
        assert len(dims) >= 2

        dim_in = dims[0]
        dim_out = dims[-1]
        dim_hiddens = list(dims[1:-1])

        self.dim_out = dim_out

        self.output = np.empty((pop_size, self.dim_out), dtype = np.float32)

        self.top = Topology(dim_in, dim_out, num_hiddens = dim_hiddens, pop_size = pop_size, mutation_hyper_params = mutation_hyper_params, crossover_hyper_params = crossover_hyper_params, selection_hyper_params = selection_hyper_params, num_islands = num_islands)
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
