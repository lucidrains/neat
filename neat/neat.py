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
    add_node,
    add_edge,
    tournament,
    crossover,
    select,
    mutate,
    evaluate_nn
)

# functions

def add_neat_topology():
    return add_topology()

def add_neuron(top_id):
    return add_node(top_id)

def add_synapse(top_id, from_id, to_id):
    return add_edge(top_id, from_id, to_id)

def generate_hyper_weight(
    top_id,
    nn_id,
    shape: tuple[int, ...]
) -> Array:

    raise NotImplementedError

# mlp for actor

def init_mlp_weights_biases(
    *dims,
    pop_dim = 1,
    add_pop_dim = True,
    seed = 42
):
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

    return weights, biases

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

# classes

def genetic_algorithm_step(
    fitnesses: Array,
    policy_weights: list[Array],
    policy_biases: list[Array],
    top_id: int = 0,
    num_selected = 2
):
    assert num_selected >= 2

    # todo
    # 1. selection
    # 2. tournament -> parent pairs
    # 3. compute children with crossover
    # 4. concat children to population
    # 5. mutation

    selected_indices = select(top_id, fitnesses.tolist(), num_selected)

    return policy_weights, policy_biases
