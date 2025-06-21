import jax

import einx
from einops import rearrange

import nimporter
from neat.neat_nim import add

# functions

def nim_add(x, y):
    return add(x, y)

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
