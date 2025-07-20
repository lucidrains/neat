import numpy as np
from random import randrange
from tqdm import tqdm

import jax.numpy as jnp

from neat.neat import HyperNEAT, NEAT

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# constants

NUM_GENERATIONS = 1000

TEST_REGULAR_NEAT = True
POP_SIZE = 100
NUM_CPPN_HIDDEN_NODES = 12
NUM_HIDDEN_LAYERS = 0

RECORD_EVERY = 10
MAX_EPISODE_LEN = 250
FRAC_NATURAL_SELECTED = 0.25
TOURNAMENT_FRAC = 0.25
NUM_ROLLOUTS_BEFORE_EVO = 1

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
    seed = None
):

    state, _ = env.reset(seed = seed)

    while True:
        actions_to_env = population.single_forward(policy_index, state, sample = True)

        next_state, reward, truncated, terminated, *_ = env.step(actions_to_env)

        state = next_state

        if truncated or terminated:
            break

    env.close()

# population policies

dim_state = envs.observation_space.shape[-1]
num_actions = 4

if TEST_REGULAR_NEAT:
    population = NEAT(
        dim_state, *((NUM_CPPN_HIDDEN_NODES,) * NUM_HIDDEN_LAYERS), num_actions,
        pop_size = POP_SIZE
    )
else:
    population = HyperNEAT(
        dim_state, 16, 16, num_actions,

        num_hiddens = (NUM_CPPN_HIDDEN_NODES,) * NUM_HIDDEN_LAYERS,
        pop_size = POP_SIZE
    )

# interact with environment across generations

for gen in tqdm(range(NUM_GENERATIONS)):

    all_fitnesses = []
    seed = randrange(int(1e7))

    for _ in range(NUM_ROLLOUTS_BEFORE_EVO):

        state, _ = envs.reset(seed = seed)

        done = None
        time = 0
        rewards = []

        while True:
            actions_to_env = population.forward(state, sample = True)

            next_state, reward, truncated, terminated, *_ = envs.step(actions_to_env)

            # gymnasium should just make terminated always True if one env terminates before the other..

            is_done_this_step = truncated | terminated
            done = default(done, is_done_this_step)
            done |= is_done_this_step

            step_reward = reward * done.astype(jnp.float32)  # insurance, in case gymnasium borks and returns rewards for terminated envs in a collection of vec envs

            rewards.append(step_reward)
            state = next_state

            if time >= MAX_EPISODE_LEN:
                break

            if done.all():
                break

            time += 1

        env.close()

        rewards = jnp.stack(rewards)

        one_rollout_fitnesses = rewards.sum(axis = 0) # cumulative rewards as fitnesses

        all_fitnesses.append(one_rollout_fitnesses)

    fitnesses = jnp.stack(all_fitnesses).mean(axis = 0)

    # insilico evolution

    population.genetic_algorithm_step(
        fitnesses,
        num_selected_frac = FRAC_NATURAL_SELECTED,
        tournament_frac = TOURNAMENT_FRAC
    )

    if divisible_by(gen + 1, RECORD_EVERY):
        record_agent_(0)
