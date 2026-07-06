# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "x-neat",
#     "einops",
#     "numpy>=2.2.5",
#     "tqdm",
#     "wandb[media]",
#     "fire",
#     "imageio",
#     "imageio-ffmpeg"
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
from einops import rearrange

from tqdm import tqdm
import wandb
import imageio

from neat.neat import NEAT
from neat.neat_nim import get_population_complexities

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# environment

class VecSnakeEnv:
    def __init__(self, num_envs, grid_size = 4, max_steps = 40, food_reward = 1.0, hit_wall_penalty = -5.0, existence_penalty = -0.01):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self._max_steps = max_steps
        self.food_reward = food_reward
        self.hit_wall_penalty = hit_wall_penalty
        self.existence_penalty = existence_penalty
        
        self.max_len = grid_size * grid_size
        
        self.dx_map = np.array([0, 1, 0, -1], dtype=np.int32)
        self.dy_map = np.array([-1, 0, 1, 0], dtype=np.int32)

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value):
        self._max_steps = value
        
    def reset(self, seed = None):
        if exists(seed):
            np.random.seed(seed)
            import random
            random.seed(seed)
            
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.done = np.zeros(self.num_envs, dtype=bool)
        
        self.snake = np.zeros((self.num_envs, self.max_len, 2), dtype=np.int32)
        self.snake[:, 0, 0] = np.random.randint(self.grid_size, size=self.num_envs)
        self.snake[:, 0, 1] = np.random.randint(self.grid_size, size=self.num_envs)
        self.snake_length = np.ones(self.num_envs, dtype=np.int32)
        
        self.direction = np.random.randint(4, size=self.num_envs)
        
        self.food = np.zeros((self.num_envs, 2), dtype=np.int32)
        mask = np.ones(self.num_envs, dtype=bool)
        self._place_food_vectorized(mask)
        
        return self._extract_obs(), dict()

    def _place_food_vectorized(self, mask):
        num_needs_food = mask.sum()
        if num_needs_food == 0:
            return
            
        needs_food_indices = np.where(mask)[0]
        
        placed = np.zeros(num_needs_food, dtype=bool)
        new_food = np.zeros((num_needs_food, 2), dtype=np.int32)
        
        while not placed.all():
            unplaced_idx = np.where(~placed)[0]
            count = len(unplaced_idx)
            env_idxs = needs_food_indices[unplaced_idx]
            
            cand_x = np.random.randint(self.grid_size, size=count)
            cand_y = np.random.randint(self.grid_size, size=count)
            cand_pos = np.stack([cand_x, cand_y], axis=-1)
            
            match = (self.snake[env_idxs, :, 0] == cand_x[:, None]) & (self.snake[env_idxs, :, 1] == cand_y[:, None])
            valid_mask = np.arange(self.max_len)[None, :] < self.snake_length[env_idxs, None]
            
            collide = (match & valid_mask).any(axis=1)
            
            valid = ~collide
            new_food[unplaced_idx[valid]] = cand_pos[valid]
            placed[unplaced_idx] = valid
            
        self.food[mask] = new_food

    def _extract_obs(self):
        grid = np.zeros((self.num_envs, 3, self.grid_size, self.grid_size), dtype=np.float32)
        env_indices = np.arange(self.num_envs)
        
        # food
        grid[env_indices, 0, self.food[:, 1], self.food[:, 0]] = 1.0
        
        # head
        grid[env_indices, 1, self.snake[:, 0, 1], self.snake[:, 0, 0]] = 1.0
        
        # body
        for i in range(1, self.max_len):
            valid = i < self.snake_length
            valid_envs = env_indices[valid]
            if len(valid_envs) > 0:
                y = self.snake[valid_envs, i, 1]
                x = self.snake[valid_envs, i, 0]
                grid[valid_envs, 2, y, x] = 1.0
                
        dir_onehot = np.zeros((self.num_envs, 4), dtype=np.float32)
        dir_onehot[env_indices, self.direction] = 1.0
        
        obs = np.concatenate([grid.reshape(self.num_envs, -1), dir_onehot], axis=-1)
        return obs

    def step(self, actions):
        actions = np.asarray(actions)
        update_mask = ~self.done
        
        self.steps = np.where(update_mask, self.steps + 1, self.steps)
        
        valid_action = np.abs(actions - self.direction) != 2
        self.direction = np.where(valid_action & update_mask, actions, self.direction)
        
        dx = self.dx_map[self.direction]
        dy = self.dy_map[self.direction]
        
        new_head_x = self.snake[:, 0, 0] + dx
        new_head_y = self.snake[:, 0, 1] + dy
        new_head = np.stack([new_head_x, new_head_y], axis=-1)
        
        oob = (new_head_x < 0) | (new_head_x >= self.grid_size) | (new_head_y < 0) | (new_head_y >= self.grid_size)
        
        match = (self.snake[:, :, 0] == new_head_x[:, None]) & (self.snake[:, :, 1] == new_head_y[:, None])
        valid_mask = np.arange(self.max_len)[None, :] < self.snake_length[:, None]
        self_collision = (match & valid_mask).any(axis=1)
        
        terminated = oob | self_collision
        truncated = self.steps >= self._max_steps
        
        ate_food = (new_head_x == self.food[:, 0]) & (new_head_y == self.food[:, 1])
        
        new_snake = np.zeros_like(self.snake)
        new_snake[:, 1:] = self.snake[:, :-1]
        new_snake[:, 0] = new_head
        
        should_update_snake = update_mask & ~terminated
        
        self.snake = np.where(should_update_snake[:, None, None], new_snake, self.snake)
        self.snake_length = np.where(should_update_snake, self.snake_length + ate_food.astype(np.int32), self.snake_length)
        
        needs_food_mask = should_update_snake & ate_food
        self._place_food_vectorized(needs_food_mask)
        
        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward = np.where(terminated, self.hit_wall_penalty,
                 np.where(ate_food, self.food_reward, self.existence_penalty))
        
        reward = np.where(self.done, 0.0, reward)
        
        out_terminated = np.where(self.done, True, terminated)
        out_truncated = np.where(self.done, False, truncated)
        
        self.done = self.done | terminated | truncated
        
        obs = self._extract_obs()
        return obs, reward, out_terminated, out_truncated, dict()

    def render(self, env_idx=0):
        img = np.zeros((3, 256, 256), dtype = np.float32)
        c = 256 // self.grid_size
        fx, fy = self.food[env_idx]

        img[0, fy*c:(fy+1)*c, fx*c:(fx+1)*c] = 1. # red for apple

        for i in range(self.snake_length[env_idx]):
            sx, sy = self.snake[env_idx, i]
            y1, y2, x1, x2 = sy*c, (sy+1)*c, sx*c, (sx+1)*c
            img[1, y1:y2, x1:x2] = 1. if i == 0 else 0.8 # green for snake

            if i == 0:
                direction = self.direction[env_idx]
                if   direction == 0: slice_y, slice_x = slice(y1, y1+1), slice(x1, x2)
                elif direction == 1: slice_y, slice_x = slice(y1, y2), slice(x2-1, x2)
                elif direction == 2: slice_y, slice_x = slice(y2-1, y2), slice(x1, x2)
                elif direction == 3: slice_y, slice_x = slice(y1, y2), slice(x1, x1+1)
                img[:, slice_y, slice_x] = 1.

        return img

# main training script

def train(
    num_generations: int = 2000,
    pop_size: int = 1000,

    # intervals
    record_every: int = 25,
    save_population_every: int = 100,

    # environment parameters
    start_max_episode_len: int = 40,
    end_max_episode_len: int = 40,
    curriculum_generations: int = 200,
    num_rollouts_before_evo: int = 2,
    num_recurrent: int = 0,

    # rewards
    food_reward: float = 1.0,
    hit_wall_penalty: float = -5.0,
    existence_penalty: float = -0.01,

    # selection parameters
    frac_natural_selected: float = 0.2,
    tournament_size: int = 10,
    prob_weigh_complexity_as_fitness: float = 0.05,
    use_fuss: bool = False,
    fuss_eps: float = 1e-5,

    # island parameters
    num_islands: int = 4,
    migrate_every: int = 25,
    migrate_num: int = 5,
    reset_islands_every: int = 100,
    reset_islands_num: int = 0,

    # mutation parameters
    mutate_prob: float = 0.95,
    use_self_adaptive_mutation: bool = False,
    use_fast_ga: bool = False,
    fast_ga_beta: float = 1.5,
    add_novel_edge_prob: float = 5e-5,
    toggle_meta_edge_prob: float = 0.05,
    add_remove_node_prob: float = 1e-5,
    change_activation_prob: float = 0.02,
    change_edge_weight_prob: float = 0.1,
    replace_edge_weight_prob: float = 0.2,
    change_node_bias_prob: float = 0.1,
    replace_node_bias_prob: float = 0.2,
    grow_edge_prob: float = 5e-6,
    grow_node_prob: float = 1e-5,
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
    recording_folder: str = './recordings-snake',
    recorded_population_folder: str = './recorded-populations-snake',
    use_wandb: bool = False,
):

    print(f'\nrecordings will be saved to {Path(recording_folder).resolve()}, every {record_every} generations\n')

    run_name = f'neat-snake-{pop_size}'

    # hyperparams dictionaries

    selection_hyper_params = dict(
        frac_natural_selected = frac_natural_selected,
        tournament_size = tournament_size,
        use_fuss = use_fuss,
        fuss_eps = fuss_eps
    )

    mutation_hyper_params = dict(
        mutate_prob = mutate_prob,
        use_self_adaptive_mutation = use_self_adaptive_mutation,
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
        num_recurrent = num_recurrent,
    )

    # initialize experiment tracker

    wandb.init(project = 'snake-neat', mode = 'online' if use_wandb else 'disabled')
    wandb.run.name = run_name

    # environments

    envs = VecSnakeEnv(
        num_envs = pop_size,
        grid_size = 4,
        max_steps = end_max_episode_len,
        food_reward = food_reward,
        hit_wall_penalty = hit_wall_penalty,
        existence_penalty = existence_penalty
    )

    # single environment for recording

    rmtree(recording_folder, ignore_errors = True)
    rmtree(recorded_population_folder, ignore_errors = True)
    Path(recording_folder).mkdir(parents = True, exist_ok = True)
    Path(recorded_population_folder).mkdir(parents = True, exist_ok = True)

    num_recorded = 0

    def record_agent_(policy_index, seed = None):
        nonlocal num_recorded

        population.reset_recurrent_state()
        rec_env = VecSnakeEnv(
            num_envs = 1, grid_size = 4, max_steps = end_max_episode_len,
            food_reward = food_reward, hit_wall_penalty = hit_wall_penalty, existence_penalty = existence_penalty
        )

        state, _ = rec_env.reset(seed = seed)
        state = state[0]
        
        frames = []

        while True:
            img = rec_env.render(0)
            frames.append((rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8))
            
            actions_to_env = population.single_forward(policy_index, state, sample = True, temperature = 0.)

            next_state, reward, terminated, truncated, *_ = rec_env.step([actions_to_env])
            state = next_state[0]

            if truncated[0] or terminated[0]:
                img = rec_env.render(0)
                frames.append((rearrange(img, 'c h w -> h w c') * 255).astype(np.uint8))
                break

        video_path = f'{recording_folder}/snake-video-episode-{num_recorded}.mp4'
        imageio.mimwrite(video_path, frames, fps = 8, macro_block_size = 1)

        video = wandb.Video(video_path, format = 'mp4')
        wandb.log(dict(fittest_rollout = video))

    # set up population

    dim_state = 3 * 16 + 4 # 3 channels (food, head, body) + 4-dim direction one-hot
    num_actions = 4

    population = NEAT(
        dim_state, 16, num_actions,
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
        
        envs.max_steps = current_max_episode_len

        for _ in range(num_rollouts_before_evo):
            population.reset_recurrent_state()
            state, _ = envs.reset(seed = seed)

            done = np.zeros(pop_size, dtype = bool)
            reward_buffer[:] = 0.
            time = 0

            while True:
                actions_to_env = population.forward(state, sample = True, temperature = 0.)

                next_state, reward, terminated, truncated, *_ = envs.step(actions_to_env)
                is_done_this_step = truncated | terminated

                reward_buffer[time] = reward * (~done).astype(np.float32)
                done |= is_done_this_step

                state = next_state
                time += 1

                if time >= current_max_episode_len or done.all():
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
