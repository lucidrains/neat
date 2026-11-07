# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "x-neat",
#     "gymnasium[box2d]",
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "numpy",
#     "fire",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# x-neat = { path = "." }
# ///

"""Baldwinian NEAT on windy LunarLander or nonstationary Pendulum.

    frozen : evolve [x, m] with m pinned at zero
    ttt    : evolve [x, m] with m updated by predictive-coding gradients

    .venv/bin/python train_baldwin.py --quick
    .venv/bin/python train_baldwin.py --env lunar --noise 0.05
    .venv/bin/python train_baldwin.py --env pendulum
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from neat import BaldwinNEAT
from neat.neat import set_seed


def exists(v):
    return v is not None

SELECTION = dict(frac_natural_selected = 0.15, tournament_size = 4, use_fuss = False, fuss_eps = 1e-5)

MUTATION = dict(
    mutate_prob = 0.95,
    use_self_adaptive_mutation = False,
    use_fast_ga = False,
    fast_ga_beta = 1.5,
    add_novel_edge_prob = 0.05,
    toggle_meta_edge_prob = 0.05,
    add_remove_node_prob = 1e-5,
    change_activation_prob = 0.001,
    change_edge_weight_prob = 0.5,
    replace_edge_weight_prob = 0.2,
    change_node_bias_prob = 0.1,
    replace_node_bias_prob = 0.2,
    grow_edge_prob = 1e-3,
    grow_node_prob = 5e-4,
    perturb_weight_strength = 0.2,
    perturb_bias_strength = 0.2,
    num_preserve_elites = 2,
    max_weight_magnitude = 5.0
)

CROSSOVER = dict(
    prob_child_disabled_given_parent_cond = 0.75,
    prob_remove_disabled_node = 0.01,
    prob_inherit_all_excess_genes = 1.0
)

COLORS = dict(frozen = '#0ea5e9', ttt = '#10b981', ttt_carry = '#8b5cf6')

# config

@dataclass
class Config:
    env: str = 'lunar'
    modes: tuple[str, ...] = ('frozen', 'ttt')
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pop_size: int = 64
    generations: int = 40
    episodes: int = 2
    memory: int = 4
    memory_lr: float = 0.05
    noise: float = 0.0
    heldout: int = 6
    env_seed: int = 20_000
    output: str = 'baldwin'

    def __post_init__(self):
        assert self.env in ('lunar', 'pendulum'), f'unknown env {self.env!r}'
        assert all(mode in COLORS for mode in self.modes)

    @property
    def full_obs_dim(self) -> int:
        return 8 if self.env == 'lunar' else 3

    @property
    def action_dim(self) -> int:
        return 2 if self.env == 'lunar' else 1

    @property
    def horizon(self) -> int:
        return 450 if self.env == 'lunar' else 200

    @property
    def heldout_horizon(self) -> int:
        return 600 if self.env == 'lunar' else 200

# environments

class NonstationaryPendulum(gym.Wrapper):
    """Pendulum with gravity and pole length switching every 50 steps."""

    def __init__(self):
        super().__init__(gym.make('Pendulum-v1'))
        self.steps = 0
        self.schedule_rng = None

    def _switch(self):
        self.unwrapped.g = float(self.schedule_rng.uniform(5.0, 15.0))
        self.unwrapped.l = float(self.schedule_rng.uniform(0.5, 1.5))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.schedule_rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))
        self.steps = 0
        self._switch()
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        if self.steps % 50 == 0:
            self._switch()
        return result

def make_envs(cfg: Config):
    if cfg.env == 'lunar':
        return gym.make_vec(
            'LunarLanderContinuous-v3',
            num_envs = cfg.pop_size,
            enable_wind = True,
            wind_power = 10.0,
            turbulence_power = 1.0,
            vectorization_mode = 'sync'
        )

    return gym.vector.SyncVectorEnv([NonstationaryPendulum for _ in range(cfg.pop_size)])

# rollout

def rollout(pop, envs, cfg: Config, seed: int, *, horizon: int, episodes: int):
    size = len(pop.output)
    returns = np.zeros(size)
    noise_rng = np.random.default_rng(seed)
    low = np.asarray(envs.single_action_space.low, dtype = np.float32)
    high = np.asarray(envs.single_action_space.high, dtype = np.float32)

    for episode in range(episodes):
        pop.reset_episode()
        obs, _ = envs.reset(seed = seed + episode)

        done = np.zeros(size, dtype = bool)
        episode_returns = np.zeros(size)

        for _ in range(horizon):
            if cfg.noise > 0.0:
                obs = (obs + noise_rng.normal(0.0, cfg.noise, size = obs.shape)).astype(np.float32)

            actions = np.clip(pop.forward(obs), low, high)
            next_obs, reward, terminated, truncated, _ = envs.step(actions)
            step_done = terminated | truncated

            alive = ~done
            episode_returns[alive] += reward[alive]
            pop.observe_transition(obs, next_obs, dones = step_done)

            done |= step_done
            obs = next_obs

            if done.all():
                break

        returns += episode_returns

    return returns / episodes

def evaluate(pop, envs, cfg: Config, seed: int):
    rows = [
        rollout(pop, envs, cfg, seed + episode * 1_000, horizon = cfg.heldout_horizon, episodes = 1)
        for episode in range(cfg.heldout)
    ]
    return np.stack(rows)

# one run

def run(cfg: Config, mode: str, seed: int):
    set_seed(seed)
    envs = make_envs(cfg)

    try:
        pop = BaldwinNEAT(
            cfg.full_obs_dim,
            cfg.action_dim,
            pop_size = cfg.pop_size,
            num_memory = cfg.memory,
            memory_lr = cfg.memory_lr,
            ttt = mode != 'frozen',
            carry_across_episodes = mode == 'ttt_carry',
            memory_seed = seed,
            mutation_hyper_params = MUTATION,
            crossover_hyper_params = CROSSOVER,
            selection_hyper_params = SELECTION
        )
        pop.reset_recurrent_state()

        best, topq = [], []
        for generation in tqdm(range(cfg.generations), desc = f'{mode} seed={seed}', leave = False):
            fitnesses = rollout(
                pop, envs, cfg,
                cfg.env_seed + seed * 100_000 + generation,
                horizon = cfg.horizon,
                episodes = cfg.episodes
            )
            best.append(float(fitnesses.max()))
            topq.append(float(np.sort(fitnesses)[-max(1, cfg.pop_size // 4):].mean()))

            if generation < cfg.generations - 1:
                pop.genetic_algorithm_step(
                    fitnesses,
                    selection_hyper_params = SELECTION,
                    mutation_hyper_params = MUTATION,
                    crossover_hyper_params = CROSSOVER
                )

        elite = np.argsort(fitnesses)[-max(1, cfg.pop_size // 4):]
        heldout_seed = cfg.env_seed + seed * 100_000 + 900_000

        pop.set_ttt(True)
        on = evaluate(pop, envs, cfg, heldout_seed)[:, elite].mean(axis = 1)
        pop.set_ttt(False)
        off = evaluate(pop, envs, cfg, heldout_seed)[:, elite].mean(axis = 1)

        return dict(
            mode = mode,
            seed = seed,
            best = best,
            topq = topq,
            on = on.tolist(),
            off = off.tolist(),
            natural = (on if mode != 'frozen' else off).tolist()
        )
    finally:
        envs.close()

# summary

def sem(x):
    return float(x.std(ddof = 1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

def summarize(results, cfg: Config):
    summary = {}

    for mode in cfg.modes:
        rows = [r for r in results if r['mode'] == mode]
        if not rows:
            continue

        topq = np.array([r['topq'] for r in rows])
        on = np.array([r['on'] for r in rows])
        off = np.array([r['off'] for r in rows])
        natural = np.array([r['natural'] for r in rows])
        late = min(5, topq.shape[1])

        summary[mode] = dict(
            n = len(rows),
            late_topq = float(topq[:, -late:].mean()),
            late_topq_sem = sem(topq[:, -late:].mean(axis = 1)),
            on = float(on.mean()),
            off = float(off.mean()),
            natural = float(natural.mean()),
            ttt_gain = float((on - off).mean()),
            ttt_gain_sem = sem((on - off).mean(axis = 1))
        )

    baselines = [m for m in cfg.modes if m != 'ttt' and m in summary]
    if 'ttt' in summary and baselines:
        base_mode = baselines[0]
        base = {r['seed']: r for r in results if r['mode'] == base_mode}
        ttt = {r['seed']: r for r in results if r['mode'] == 'ttt'}
        seeds = sorted(set(base) & set(ttt))

        base_topq = np.array([np.mean(base[s]['topq'][-5:]) for s in seeds])
        ttt_topq = np.array([np.mean(ttt[s]['topq'][-5:]) for s in seeds])
        base_out = np.array([np.mean(base[s]['natural']) for s in seeds])
        ttt_out = np.array([np.mean(ttt[s]['natural']) for s in seeds])

        summary['comparison'] = dict(
            base_mode = base_mode,
            n = len(seeds),
            topq_base = float(base_topq.mean()),
            topq_ttt = float(ttt_topq.mean()),
            topq_wins = int((ttt_topq > base_topq).sum()),
            topq_p = paired_p(ttt_topq, base_topq),
            out_base = float(base_out.mean()),
            out_ttt = float(ttt_out.mean()),
            out_wins = int((ttt_out > base_out).sum()),
            out_p = paired_p(ttt_out, base_out)
        )

    return summary

def paired_p(a, b, n = 20_000, seed = 0):
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size = (n, len(diff)))
    observed = abs(diff.mean())
    return float((np.abs((signs * diff).mean(axis = 1)) >= observed).mean())

# plot

def plot(results, summary, cfg: Config):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError:
        return

    sns.set_theme(style = 'whitegrid', context = 'talk')

    curves = pd.DataFrame([
        dict(mode = r['mode'], seed = r['seed'], generation = i + 1, best = b, topq = t)
        for r in results
        for i, (b, t) in enumerate(zip(r['best'], r['topq']))
    ])
    heldout = pd.DataFrame([
        dict(mode = r['mode'], seed = r['seed'], return_ = np.mean(r['natural']))
        for r in results
    ])

    fig, axes = plt.subplots(2, 2, figsize = (14, 10), dpi = 160)

    for ax, key, title in ((axes[0, 0], 'topq', 'Top-quartile fitness'), (axes[0, 1], 'best', 'Best fitness')):
        sns.lineplot(data = curves, x = 'generation', y = key, hue = 'mode', palette = COLORS, errorbar = 'se', ax = ax, lw = 2.2)
        ax.set_title(title)
        ax.set_ylabel('return')
        ax.legend(title = None)

    sns.barplot(data = heldout, x = 'mode', y = 'return_', palette = COLORS, errorbar = 'se', capsize = 0.12, alpha = 0.8, ax = axes[1, 0], legend = False)
    sns.stripplot(data = heldout, x = 'mode', y = 'return_', color = 'black', size = 5, alpha = 0.7, ax = axes[1, 0])
    axes[1, 0].axhline(0.0, color = 'black', lw = 0.8)
    axes[1, 0].set_title('Held-out return')
    axes[1, 0].set_xlabel('')

    comparison = summary.get('comparison')
    if exists(comparison):
        base = {r['seed']: np.mean(r['natural']) for r in results if r['mode'] == comparison['base_mode']}
        ttt = {r['seed']: np.mean(r['natural']) for r in results if r['mode'] == 'ttt'}
        gains = pd.DataFrame([
            dict(label = f'ttt - {comparison["base_mode"]}', gain = ttt[s] - base[s])
            for s in sorted(set(base) & set(ttt))
        ])
        sns.barplot(data = gains, x = 'label', y = 'gain', color = '#10b981', errorbar = 'se', capsize = 0.12, alpha = 0.8, ax = axes[1, 1], legend = False)
        sns.stripplot(data = gains, x = 'label', y = 'gain', color = 'black', size = 5, alpha = 0.7, ax = axes[1, 1])
        axes[1, 1].axhline(0.0, color = 'black', lw = 0.8)
        axes[1, 1].set_title('Paired held-out gain')
        axes[1, 1].set_xlabel('')

    sns.despine(fig)
    fig.suptitle(f'Baldwinian NEAT on {cfg.env}', fontweight = 'bold')
    fig.tight_layout(rect = [0, 0, 1, 0.96])
    fig.savefig(Path(cfg.output).with_suffix('.png'))
    plt.close(fig)

# entry point

def train_baldwin(
    env: str = 'lunar',
    modes: str = 'frozen,ttt',
    seeds: str = '0,1,2,3,4',
    pop_size: int = 64,
    generations: int = 40,
    episodes: int = 2,
    memory: int = 4,
    memory_lr: float = 0.05,
    noise: float = 0.0,
    heldout: int = 6,
    env_seed: int = 20_000,
    output: str = 'baldwin',
    quick: bool = False
):
    if quick:
        pop_size, generations, episodes, heldout, seeds = 16, 4, 1, 2, '0'

    def parse(value):
        items = value if isinstance(value, (list, tuple)) else str(value).split(',')
        return tuple(str(item).strip() for item in items if str(item).strip())

    cfg = Config(
        env = env,
        modes = parse(modes),
        seeds = tuple(int(s) for s in parse(seeds)),
        pop_size = pop_size,
        generations = generations,
        episodes = episodes,
        memory = memory,
        memory_lr = memory_lr,
        noise = noise,
        heldout = heldout,
        env_seed = env_seed,
        output = output
    )

    print(json.dumps(asdict(cfg), indent = 2))

    start = time.time()
    results = [run(cfg, mode, seed) for seed in cfg.seeds for mode in cfg.modes]
    summary = summarize(results, cfg)

    Path(cfg.output).with_suffix('.json').write_text(
        json.dumps(dict(config = asdict(cfg), results = results, summary = summary, elapsed = time.time() - start), indent = 2)
    )

    print(f'\ncompleted in {time.time() - start:.1f}s')
    for mode in cfg.modes:
        s = summary[mode]
        print(f'{mode:10s} | late topQ {s["late_topq"]:7.2f} ± {s["late_topq_sem"]:5.2f} | '
              f'held-out {s["natural"]:7.2f} | TTT gain {s["ttt_gain"]:+6.2f}')

    comparison = summary.get('comparison')
    if exists(comparison):
        c = comparison
        print(f'\nTTT vs {c["base_mode"]} (late topQ): {c["topq_ttt"]:.2f} vs {c["topq_base"]:.2f} | '
              f'{c["topq_wins"]}/{c["n"]} seeds | p={c["topq_p"]:.4f}')
        print(f'TTT vs {c["base_mode"]} (held-out): {c["out_ttt"]:.2f} vs {c["out_base"]:.2f} | '
              f'{c["out_wins"]}/{c["n"]} seeds | p={c["out_p"]:.4f}')

    plot(results, summary, cfg)

if __name__ == '__main__':
    fire.Fire(train_baldwin)
