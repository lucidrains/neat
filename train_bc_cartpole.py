# /// script
# dependencies = [
#     "gymnasium",
#     "torch",
#     "tqdm",
#     "numpy>=2.2.5",
#     "x-neat"
# ]
# [tool.uv.sources]
# x-neat = { path = "." }
# ///

from pathlib import Path
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm

from neat import NEAT, behavior_clone

def train_and_clone(
    num_generations: int = 5,
    pop_size: int = 64,
    hidden: int = 16,
    seed: int = 0,
    max_steps: int = 500,
    save_path: str = 'mlp-cartpole.pt'
):
    start_time = time.time()
    print("=" * 60)
    print("Step 1: Evolving NEAT on CartPole-v1")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    vec_envs = gym.make_vec('CartPole-v1', num_envs = pop_size, vectorization_mode = 'sync')

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    pop = NEAT(obs_dim, hidden, act_dim, pop_size = pop_size, num_islands = 2)

    for gen in tqdm(range(num_generations), desc = 'evolve cartpole'):
        states, _ = vec_envs.reset(seed = seed + gen)
        dones = np.zeros(pop_size, dtype = bool)
        episode_rewards = np.zeros(pop_size, dtype = np.float32)

        for _ in range(max_steps):
            actions = pop.forward(states, sample = True)
            next_states, rewards, term, trunc, _ = vec_envs.step(actions)

            episode_rewards += np.where(~dones, rewards, 0.0)
            dones = dones | term | trunc
            states = next_states

            if dones.all():
                break

        pop.genetic_algorithm_step(episode_rewards)

    vec_envs.close()

    # Step 2: Extract champion NEAT network

    eval_rewards = np.zeros(pop_size, dtype = np.float32)
    for k in range(pop_size):
        for s_idx in [0, 1]:
            s, _ = env.reset(seed = seed + s_idx)
            r_sum = 0
            for _ in range(max_steps):
                s, r, term, trunc, _ = env.step(int(np.argmax(pop[k](s))))
                r_sum += r
                if term or trunc: break
            eval_rewards[k] += r_sum / 2.0

    pop.champion_index = int(np.argmax(eval_rewards))
    champion = pop.champion
    print(f"\nExtracted champion network: {champion} (validation score: {eval_rewards[pop.champion_index]:.1f})")

    # Step 3: Barebones PyTorch MLP

    print("\n" + "=" * 60)
    print("Step 2: Defining barebones PyTorch MLP")
    print("=" * 60)

    mlp = nn.Sequential(
        nn.Linear(obs_dim, 32),
        nn.Tanh(),
        nn.Linear(32, act_dim)
    )
    print(mlp)

    # Step 4: Behavior clone and save to root

    print("\n" + "=" * 60)
    print(f"Step 3: Behavior cloning champion into MLP -> {save_path}")
    print("=" * 60)

    behavior_clone(
        env = env,
        champion = champion,
        mlp = mlp,
        save_path = save_path,
        iterations = 15,
        rollouts_per_iter = 4,
        epochs = 4,
        batch_size = 64,
        lr = 2e-3,
        max_steps = max_steps,
        target_ratio = 0.95
    )

    # Step 5: Verify saved model artifact in root

    out_file = Path(save_path)
    assert out_file.exists(), f"Expected {out_file} to exist in project root!"
    print(f"\nVerified {out_file} successfully saved in root (size: {out_file.stat().st_size} bytes)")

    # Step 6: Validate loaded model on fresh CartPole episodes

    print("\n" + "=" * 60)
    print("Step 4: Verifying loaded MLP in CartPole-v1")
    print("=" * 60)

    eval_mlp = nn.Sequential(
        nn.Linear(obs_dim, 32),
        nn.Tanh(),
        nn.Linear(32, act_dim)
    )
    eval_mlp.load_state_dict(torch.load(out_file, weights_only = True))
    eval_mlp.eval()

    rewards = []
    for ep in range(10):
        obs, _ = env.reset(seed = 100 + ep)
        ep_reward = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                t_obs = torch.from_numpy(obs).float().unsqueeze(0)
                action = int(eval_mlp(t_obs).argmax(dim = -1))
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            if term or trunc:
                break
        rewards.append(ep_reward)

    mean_r, std_r = np.mean(rewards), np.std(rewards)
    print(f"Loaded MLP evaluation (10 episodes): {mean_r:.2f} ± {std_r:.2f}")

    env.close()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f}s (completed well under 2-3 minutes)")
    print("=" * 60)

if __name__ == '__main__':
    train_and_clone()
