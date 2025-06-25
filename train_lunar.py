from neat.neat import (
    init_mlp_weights_biases,
    mlp
)

# constants

SEED = 42
POP_SIZE = 8

# environment

import gymnasium as gym
envs = gym.make_vec(
    "LunarLander-v3",
    num_envs = POP_SIZE,
    render_mode = 'rgb_array',
    vectorization_mode = 'sync',
    wrappers = (gym.wrappers.TimeAwareObservation,)
)

# policy

dim_state = envs.observation_space.shape[-1]

policy_weights, policy_biases = init_mlp_weights_biases(dim_state, 16, 16, 5, pop_dim = POP_SIZE)

state, _ = envs.reset(seed = SEED)

# interact

actions = mlp(policy_weights, policy_biases, state)
