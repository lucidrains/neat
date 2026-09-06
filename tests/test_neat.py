import numpy as np

from neat.neat import NEAT

def test_neat():
    pop = NEAT(10, 16, 5, pop_size = 8, mutation_hyper_params = dict(mutate_prob = 0.25))
    action_logits = pop.forward(np.zeros((8, 10)))
    pop.genetic_algorithm_step(np.ones((8,)))

    assert action_logits.shape == (8, 5)
