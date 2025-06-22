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
from einops import rearrange

import nimporter
from neat.neat_nim import add_topology, add_node, add_edge

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
    seed = 42
):
    assert len(dims) > 1

    weights = []
    biases = []

    for dim_in, dim_out in zip(dims[:-1], dims[1:]):

        weights.append(jnp.zeros((dim_in, dim_out)))
        biases.append(jnp.zeros(dim_out))

    return weights, biases

@jit
def mlp(
    weights: list[Array],
    biases: list[Array],
    t: Array
):
    weights_biases = list(zip(weights, biases))

    for weight, bias in weights_biases[:-1]:
        t = t @ weight + bias
        t = nn.silu(t)

    # last layer

    weight, bias = weights_biases[-1]
    return t @ weight + bias

# classes

class GeneticAlgorithm:
    def __init__(self):
        raise NotImplementedError
