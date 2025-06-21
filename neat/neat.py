import jax

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
