# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[mujoco]",
#   "mujoco",
#   "numpy>=2.2.5",
#   "torch",
#   "torch-einops-utils",
#   "tqdm",
#   "x-mlps-pytorch",
#   "x-neat"
# ]
# [tool.uv.sources]
# x-neat = { path = "." }
# ///

# end-to-end validation:
#   discrete   : evolve -> categorical distillation
#   continuous : evolve (with exploration) -> gaussian distillation

from __future__ import annotations

import fire
import nimporter_plus
nimporter_plus.compiler_args.append('-d:danger')

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from neat.neat import NEAT
from distill_neat_to_mlp import (
    MLP,
    distill_neat_to_mlp,
    evaluate_mlp,
    evaluate_neat,
    reward_ratio,
)

# evolution

def evolve(env_name, num_gens, pop_size, hidden, exploration_noise, seed):
    env = gym.make(env_name)
    vec_envs = gym.make_vec(env_name, num_envs = pop_size, vectorization_mode = 'sync')
    obs_dim = int(env.observation_space.shape[0])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    act_dim = int(env.action_space.n) if discrete else int(env.action_space.shape[0])
    max_steps = int(env.spec.max_episode_steps)

    pop = NEAT(obs_dim, hidden, act_dim, pop_size = pop_size, num_islands = 2)

    pbar = tqdm(range(num_gens), desc = f'evolve {env_name}')
    for gen in pbar:
        states, _ = vec_envs.reset(seed = seed + gen)
        dones = np.zeros(pop_size, dtype = bool)
        episode_rewards = np.zeros(pop_size, dtype = np.float32)

        for step in range(max_steps):
            actions = pop.forward(states, sample = discrete)

            if not discrete and exploration_noise > 0.:
                actions = actions + np.random.randn(*actions.shape).astype(np.float32) * exploration_noise

            next_states, rewards, term, trunc, _ = vec_envs.step(actions)
            episode_rewards += np.where(~dones, rewards, 0.0)
            dones = dones | term | trunc
            states = next_states

            if dones.all():
                break

        pbar.set_postfix(avg = f"{episode_rewards.mean():.2f}", max = f"{episode_rewards.max():.2f}")
        pop.genetic_algorithm_step(episode_rewards)

    vec_envs.close()

    eval_rewards = np.zeros(pop_size, dtype = np.float32)
    for k in range(pop_size):
        s, _ = env.reset(seed = seed)
        done = False
        r_sum = 0.

        while not done:
            a = pop.single_forward(k, s, sample = False)
            action = int(np.argmax(a)) if discrete else np.clip(a, env.action_space.low, env.action_space.high)
            s, r, term, trunc, _ = env.step(action)
            r_sum += r
            done = term or trunc

        eval_rewards[k] = r_sum

    env.close()
    return pop, int(np.argmax(eval_rewards))

# validation

def discrete_cartpole(evo_generations = 8, pop_size = 64, distill_iterations = 100, seed = 0):
    env_name = 'CartPole-v1'
    print(f"\n{'=' * 60}\nDISCRETE: {env_name}\n{'=' * 60}")

    pop, idx = evolve(env_name, evo_generations, pop_size, 16, 0.0, seed)
    env = gym.make(env_name)

    teacher_r, _ = evaluate_neat(pop, idx, env, num_episodes = 20, start_seed = 600)
    print(f"evolved teacher: {teacher_r:.2f}")

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    mlp = MLP(obs_dim, 32, 16, act_dim)
    distill_neat_to_mlp(
        pop, idx, mlp, env,
        iterations = distill_iterations,
        rollouts_per_iter = 8,
        eval_every = distill_iterations + 1,
        max_steps = int(env.spec.max_episode_steps),
        distribution = 'categorical'
    )

    student_r, student_std = evaluate_mlp(mlp, env, num_episodes = 20, start_seed = 600, max_steps = int(env.spec.max_episode_steps))
    print(f"\nfinal: teacher {teacher_r:.2f} | mlp {student_r:.2f} ± {student_std:.2f} | ratio {reward_ratio(teacher_r, student_r):.2f}")

def continuous_inverted_pendulum(evo_generations = 120, pop_size = 128, distill_iterations = 100, seed = 0):
    env_name = 'InvertedPendulum-v4'
    print(f"\n{'=' * 60}\nCONTINUOUS: {env_name}\n{'=' * 60}")

    pop, idx = evolve(env_name, evo_generations, pop_size, 32, 0.3, seed)
    env = gym.make(env_name)

    teacher_r, teacher_std = evaluate_neat(pop, idx, env, num_episodes = 20, start_seed = 600)
    print(f"evolved teacher: {teacher_r:.2f} ± {teacher_std:.2f}")

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    mlp = MLP(obs_dim, 64, 64, 2 * act_dim)
    distill_neat_to_mlp(
        pop, idx, mlp, env,
        iterations = distill_iterations,
        rollouts_per_iter = 8,
        eval_every = distill_iterations + 1,
        max_steps = int(env.spec.max_episode_steps),
        distribution = 'continuous',
        target_clip = True
    )

    student_r, student_std = evaluate_mlp(mlp, env, num_episodes = 20, start_seed = 600, max_steps = int(env.spec.max_episode_steps), distribution = 'continuous')
    print(f"\nfinal: teacher {teacher_r:.2f} | mlp {student_r:.2f} ± {student_std:.2f} | ratio {reward_ratio(teacher_r, student_r):.2f}")

def main(
    task = 'both',  # 'discrete' | 'continuous' | 'both'
    evo_generations = None,
    distill_iterations = 100,
    pop_size = None,
    seed = 0
):
    if task in ('both', 'discrete'):
        discrete_cartpole(
            evo_generations = 8 if evo_generations is None else evo_generations,
            pop_size = 64 if pop_size is None else pop_size,
            distill_iterations = distill_iterations,
            seed = seed
        )

    if task in ('both', 'continuous'):
        continuous_inverted_pendulum(
            evo_generations = 120 if evo_generations is None else evo_generations,
            pop_size = 128 if pop_size is None else pop_size,
            distill_iterations = distill_iterations,
            seed = seed
        )

if __name__ == '__main__':
    fire.Fire(main)
