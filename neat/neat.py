import jax

import einx
from einops import rearrange

import nimporter
from neat.neat_nim import add_node, add_edge

# functions

def add_neuron():
    return add_node()

def add_synapse(from_id, to_id):
    return add_edge(from_id, to_id)

# classes

class CPPN:
    def __init__(self):
        raise NotImplementedError

class GeneticAlgorithm:
    def __init__(self):
        raise NotImplementedError

# main

if __name__ == '__main__':
    print(jax.__version__)
