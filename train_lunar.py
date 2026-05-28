# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "x-neat",
#     "gymnasium[box2d]",
#     "numpy>=2.2.5",
#     "tqdm",
#     "wandb[media]",
#     "fire",
#     "box2d-py",
#     "swig",
#     "moviepy>=1.0.3",
# ]
#
# [tool.uv.sources]
# x-neat = { path = "." }
# ///

import fire
from pathlib import Path
from shutil import rmtree
from random import randrange

import numpy as np

from tqdm import tqdm

import gymnasium as gym
import wandb

from neat.neat import NEAT
from neat.neat_nim import get_population_complexities

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main training script

def train(
    num_generations: int = 250,
    pop_size: int = 250,

    # neat configurations

    # intervals
    record_every: int = 25,
    save_population_every: int = 100,

    # environment parameters
    start_max_episode_len: int = 20,
    end_max_episode_len: int = 250,
    curriculum_generations: int = 200,
    num_rollouts_before_evo: int = 2,

    # selection parameters
    frac_natural_selected: float = 0.15,
    tournament_size: int = 3,
    prob_weigh_complexity_as_fitness: float = 0.05,

    # island parameters
    num_islands: int = 5,
    migrate_every: int = 25,
    migrate_num: int = 5,
    reset_islands_every: int = 50,
    reset_islands_num: int = 1,

    # mutation parameters
    mutate_prob: float = 0.95,
    use_fast_ga: bool = False,
    fast_ga_beta: float = 1.5,
    add_novel_edge_prob: float = 0.05,
    toggle_meta_edge_prob: float = 0.05,
    add_remove_node_prob: float = 1e-5,
    change_activation_prob: float = 0.001,
    change_edge_weight_prob: float = 0.5,
    replace_edge_weight_prob: float = 0.2,
    change_node_bias_prob: float = 0.1,
    replace_node_bias_prob: float = 0.2,
    grow_edge_prob: float = 1e-3,
    grow_node_prob: float = 5e-4,
    perturb_weight_strength: float = 0.2,
    perturb_bias_strength: float = 0.2,
    num_preserve_elites: int = 2,
    max_weight_magnitude: float = 5.0,

    # crossover parameters
    prob_child_disabled_given_parent_cond: float = 0.75,
    prob_remove_disabled_node: float = 0.01,
    prob_inherit_all_excess_genes: float = 1.0,

    # simplicity regularizer
    simplicity_weight: float = 1.0,

    # system
    recording_folder: str = './recordings',
    recorded_population_folder: str = './recorded-populations',
    wandb_online: bool = False
):

    print(f'\nrecordings will be saved to {Path(recording_folder).resolve()}, every {record_every} generations\n')

    run_name = f'neat-{pop_size}'

    # hyperparams dictionaries

    selection_hyper_params = dict(
        frac_natural_selected = frac_natural_selected,
        tournament_size = tournament_size
    )

    mutation_hyper_params = dict(
        mutate_prob = mutate_prob,
        use_fast_ga = use_fast_ga,
        fast_ga_beta = fast_ga_beta,
        add_novel_edge_prob = add_novel_edge_prob,
        toggle_meta_edge_prob = toggle_meta_edge_prob,
        add_remove_node_prob = add_remove_node_prob,
        change_activation_prob = change_activation_prob,
        change_edge_weight_prob = change_edge_weight_prob,
        replace_edge_weight_prob = replace_edge_weight_prob,
        change_node_bias_prob = change_node_bias_prob,
        replace_node_bias_prob = replace_node_bias_prob,
        grow_edge_prob = grow_edge_prob,
        grow_node_prob = grow_node_prob,
        perturb_weight_strength = perturb_weight_strength,
        perturb_bias_strength = perturb_bias_strength,
        num_preserve_elites = num_preserve_elites,
        max_weight_magnitude = max_weight_magnitude
    )

    crossover_hyper_params = dict(
        prob_child_disabled_given_parent_cond = prob_child_disabled_given_parent_cond,
        prob_remove_disabled_node = prob_remove_disabled_node,
        prob_inherit_all_excess_genes = prob_inherit_all_excess_genes
    )

    evolution_hyper_params = dict(
        mutation_hyper_params = mutation_hyper_params,
        crossover_hyper_params = crossover_hyper_params,
        selection_hyper_params = selection_hyper_params,
        num_islands = num_islands,
    )

    # initialize experiment tracker

    wandb.init(project = 'lunar-neat', mode = 'disabled' if not wandb_online else 'online')
    wandb.run.name = run_name

    # environments

    envs = gym.make_vec(
        "LunarLander-v3",
        num_envs = pop_size,
        vectorization_mode = 'sync'
    )

    # single environment for recording

    rmtree(recording_folder, ignore_errors = True)
    rmtree(recorded_population_folder, ignore_errors = True)

    rec_env = gym.make(
        "LunarLander-v3",
        render_mode = 'rgb_array'
    )

    rec_env = gym.wrappers.RecordVideo(
        env = rec_env,
        video_folder = recording_folder,
        name_prefix = 'lunar-video',
        episode_trigger = lambda eps_num: True,
        disable_logger = True
    )

    num_recorded = 0

    def record_agent_(
        policy_index,
        seed = None
    ):
        nonlocal num_recorded

        state, _ = rec_env.reset(seed = seed)

        while True:
            actions_to_env = population.single_forward(policy_index, state, sample = True)
            next_state, _, truncated, terminated, *_ = rec_env.step(actions_to_env)
            state = next_state

            if truncated or terminated:
                break

        rec_env.close()

        video = wandb.Video(
            f'{recording_folder}/lunar-video-episode-{num_recorded}.mp4',
            format = 'gif'
        )

        wandb.log(dict(fittest_rollout = video))

    # set up population

    dim_state = envs.observation_space.shape[-1]
    num_actions = 4

    population = NEAT(
        dim_state, num_actions,
        pop_size = pop_size,
        **evolution_hyper_params
    )

    # pre-allocate buffers

    reward_buffer = np.zeros((end_max_episode_len + 1, pop_size), dtype = np.float32)

    # interact with environment across generations

    pbar = tqdm(range(num_generations))

    for gen in pbar:
        all_fitnesses = np.zeros(pop_size, dtype = np.float32)
        seed = randrange(int(1e7))

        current_max_episode_len = int(np.interp(gen, [0, curriculum_generations], [start_max_episode_len, end_max_episode_len]))

        for _ in range(num_rollouts_before_evo):
            state, _ = envs.reset(seed = seed)

            done = np.zeros(pop_size, dtype = bool)
            reward_buffer[:] = 0.0
            time = 0

            while True:
                actions_to_env = population.forward(state, sample = True)
                next_state, reward, truncated, terminated, *_ = envs.step(actions_to_env)

                is_done_this_step = truncated | terminated
                reward_buffer[time] = reward * (~done).astype(np.float32)

                done |= is_done_this_step
                state = next_state
                time += 1

                if time >= current_max_episode_len:
                    break

                if done.all():
                    break

            all_fitnesses += reward_buffer[:time].sum(axis = 0)

        fitnesses = all_fitnesses / num_rollouts_before_evo

        # insilico evolution

        _migrate_num = migrate_num if divisible_by(gen + 1, migrate_every) else 0
        _reset_islands_num = reset_islands_num if divisible_by(gen + 1, reset_islands_every) else 0

        population.genetic_algorithm_step(
            fitnesses,
            selection_hyper_params = selection_hyper_params,
            mutation_hyper_params = mutation_hyper_params,
            crossover_hyper_params = crossover_hyper_params,
            migrate_num = _migrate_num,
            reset_islands_num = _reset_islands_num,
            prob_weigh_complexity_as_fitness = prob_weigh_complexity_as_fitness,
            simplicity_weight = simplicity_weight
        )

        # logging

        stats = population.stats()[0]
        nodes = stats['total_innovated_nodes']
        edges = stats['total_innovated_edges']

        mean_complexity = np.array(get_population_complexities(population.all_top_ids[0])).mean()
        max_fit = fitnesses.max()
        mean_fit = fitnesses.mean()

        wandb.log({
            'max_fitness': max_fit,
            'mean_pop_fitness': mean_fit,
            'mean_complexity': mean_complexity,
            **stats
        })

        pbar.set_description(f'C: {mean_complexity:.1f} | N: {nodes} E: {edges} | fitness: max {max_fit:.2f} | mean {mean_fit:.2f}')

        if divisible_by(gen + 1, record_every):
            record_agent_(0)
            num_recorded += 1

        if divisible_by(gen + 1, save_population_every):
            population.save_json(f'{recorded_population_folder}/population.step.{gen + 1}')

if __name__ == '__main__':
    fire.Fire(train)
