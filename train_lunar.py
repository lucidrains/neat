import numpy as np
from jax.tree_util import tree_map

from neat.neat import (
    init_mlp_weights_biases,
    mlp
)

# constants

POP_SIZE = 8

# environment

from shutil import rmtree
import gymnasium as gym

envs = gym.make_vec(
    "LunarLander-v3",
    num_envs = POP_SIZE,
    render_mode = 'rgb_array',
    vectorization_mode = 'sync'
)

# recording does not work on vectorized env, just spin up temp env for recording, and take care of picking one agent from population to record

rmtree('./recordings', ignore_errors = True)

def record_agent_(
    policy_index,
    policy_weights,
    policy_biases,
    seed = None
):
    env = gym.make(
        "LunarLander-v3",
        render_mode = 'rgb_array'
    )

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = './recordings',
        name_prefix = 'lunar-video',
        episode_trigger = lambda eps_num: True,
        disable_logger = True
    )

    single_policy_weights, single_policy_biases = tree_map(lambda t: t[policy_index], (policy_weights, policy_biases))

    state, _ = env.reset(seed = seed)

    while True:
        actions = mlp(single_policy_weights, single_policy_biases, state)

        action = np.asarray(actions.argmax(axis = -1))

        next_state, reward, truncated, terminated, *_ = env.step(action)

        state = next_state

        if truncated or terminated:
            break

    env.close()

# policy

dim_state = envs.observation_space.shape[-1]

policy_weights, policy_biases = init_mlp_weights_biases(dim_state, 16, 16, 5, pop_dim = POP_SIZE)

state, _ = envs.reset()

# interact

actions = mlp(policy_weights, policy_biases, state)

# test recording

record_agent_(0, policy_weights, policy_biases)
