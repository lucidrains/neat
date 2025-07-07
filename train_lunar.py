import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax.tree_util import tree_map

from neat.neat import PopulationMLP, mlp

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# constants

NUM_GENERATIONS = 100
POP_SIZE = 8
RECORD_EVERY = 5 # record every 5 generations

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

def record_agent_(
    policy_index,
    policy_weights,
    policy_biases,
    seed = None
):

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

# population policies

dim_state = envs.observation_space.shape[-1]

population = PopulationMLP(dim_state, 16, 16, 5, pop_size = POP_SIZE)

# interact with environment across generations

for gen in tqdm(range(NUM_GENERATIONS)):
    state, _ = envs.reset()

    rewards = []
    done = None

    while True:
        actions = population.forward(state)

        actions_to_env = np.asarray(actions.argmax(-1))
        next_state, reward, truncated, terminated, *_ = envs.step(actions_to_env)

        # gymnasium should just make terminated always True if one env terminates before the other..

        is_done_this_step = truncated | terminated
        done = default(done, is_done_this_step)
        done |= is_done_this_step

        step_reward = reward * done.astype(jnp.float32)  # insurance, in case gymnasium borks and returns rewards for terminated envs in a collection of vec envs

        rewards.append(step_reward)
        state = next_state

        if done.all():
            break

    rewards = jnp.stack(rewards)

    fitnesses = rewards.sum(axis = 0) # cumulative rewards as fitnesses

    population.genetic_algorithm_step(fitnesses)

    if divisible_by(gen + 1, RECORD_EVERY):
        record_agent_(0, population.weights, population.biases)

    print(f'cumulative rewards mean: {rewards.mean():.3f} | std: {rewards.std():.3f}')
