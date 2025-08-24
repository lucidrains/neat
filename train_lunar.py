import numpy as np
from random import randrange
from tqdm import tqdm

import jax.numpy as jnp

from neat.neat import HyperNEAT, NEAT

import wandb

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# constants

NUM_GENERATIONS = 2000

TEST_REGULAR_NEAT = True
POP_SIZE = 200
NUM_CPPN_HIDDEN_NODES = 16
NUM_HIDDEN_LAYERS = 1

RECORD_EVERY = 10
SAVE_POPULATION_EVERY = 100

MAX_EPISODE_LEN = 250
NUM_ROLLOUTS_BEFORE_EVO = 1

WANDB_ONLINE = False # turn this on to pipe experiment to cloud

RUN_NAME = f'neat-{POP_SIZE}' if TEST_REGULAR_NEAT else f'hyperneat-{POP_SIZE}'

SELECTION_HYPER_PARAMS = dict(
    frac_natural_selected = 0.25,
    tournament_frac = 0.25
)

MUTATION_HYPER_PARAMS = dict(
    mutate_prob = 0.95,
    add_novel_edge_prob = 5e-3,
    toggle_meta_edge_prob = 0.05,
    add_remove_node_prob = 1e-5,
    change_activation_prob = 0.001,
    change_edge_weight_prob = 0.5,
    replace_edge_weight_prob = 0.1,    # the percentage of time to replace the edge weight wholesale, which they did in the paper in addition to perturbing
    change_node_bias_prob = 0.1,
    replace_node_bias_prob = 0.1,
    grow_edge_prob = 5e-4,             # this is the mutation introduced in the seminal NEAT paper that takes an existing edge for a CPPN and disables it, replacing it with a new node plus two new edges. the afferent edge is initialized to 1, the efferent inherits same weight as the one disabled. this is something currently neural network frameworks simply cannot do, and what interests me
    grow_node_prob = 1e-5,             # similarly, some follow up research do a variation of the above and split an existing node into two nodes, in theory this leads to the network modularization
    perturb_weight_strength = 0.1,
    perturb_bias_strength = 0.1
)

CROSSOVER_HYPER_PARAMS = dict(
    prob_child_disabled_given_parent_cond = 0.75,
    prob_remove_disabled_node = 0.01,
    prob_inherit_all_excess_genes = 1.0
)

RECORDING_FOLDER = './recordings'
RECORDED_POPULATION_FOLDER = './recorded-populations'

# experiment tracker

wandb.init(project = 'lunar-neat', mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME

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

rmtree(RECORDING_FOLDER, ignore_errors = True)
rmtree('./recorded-population', ignore_errors = True)

env = gym.make(
    "LunarLander-v3",
    render_mode = 'rgb_array'
)

env = gym.wrappers.RecordVideo(
    env = env,
    video_folder = RECORDING_FOLDER,
    name_prefix = 'lunar-video',
    episode_trigger = lambda eps_num: True,
    disable_logger = True
)

num_recorded = 0

def record_agent_(
    policy_index,
    seed = None
):

    state, _ = env.reset(seed = seed)

    while True:
        actions_to_env = population.single_forward(policy_index, state, sample = True)

        next_state, _, truncated, terminated, *_ = env.step(actions_to_env)

        state = next_state

        if truncated or terminated:
            break

    env.close()

    video = wandb.Video(
        f'{RECORDING_FOLDER}/lunar-video-episode-{num_recorded}.mp4',
        format = 'gif'
    )

    wandb.log(dict(
        fittest_rollout = video
    ))

# population policies

dim_state = envs.observation_space.shape[-1]
num_actions = 4

evolution_hyper_params = dict(
    mutation_hyper_params = MUTATION_HYPER_PARAMS,
    crossover_hyper_params = CROSSOVER_HYPER_PARAMS,
    selection_hyper_params = SELECTION_HYPER_PARAMS,
)

if TEST_REGULAR_NEAT:
    population = NEAT(
        dim_state, *((NUM_CPPN_HIDDEN_NODES,) * NUM_HIDDEN_LAYERS), num_actions,
        pop_size = POP_SIZE,
        **evolution_hyper_params
    )
else:
    population = HyperNEAT(
        dim_state, 16, 16, num_actions,
        num_hiddens = (NUM_CPPN_HIDDEN_NODES,) * NUM_HIDDEN_LAYERS,
        pop_size = POP_SIZE,
        **evolution_hyper_params
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

        envs.close()

        rewards = jnp.stack(rewards)

        one_rollout_fitnesses = rewards.sum(axis = 0) # cumulative rewards as fitnesses

        all_fitnesses.append(one_rollout_fitnesses)

    fitnesses = jnp.stack(all_fitnesses).mean(axis = 0)

    # insilico evolution

    population.genetic_algorithm_step(fitnesses)

    # logging

    log = dict(
        max_fitness = fitnesses.max(),
        mean_pop_fitness = fitnesses.mean(),
    )

    if TEST_REGULAR_NEAT:
        stats = population.stats()[0]
        print(f"total nodes {stats['total_innovated_nodes']} | total edges: {stats['total_innovated_edges']}")

        log.update(stats)

    wandb.log(log)

    print(f'fitness: max {fitnesses.max():.2f} | mean {fitnesses.mean():.2f} | std {fitnesses.std():.2f}')

    if divisible_by(gen + 1, RECORD_EVERY):
        record_agent_(0)
        num_recorded += 1


    if divisible_by(gen + 1, SAVE_POPULATION_EVERY):
        population.save_json(f'{RECORDED_POPULATION_FOLDER}/population.step.{gen + 1}')
