# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "x-neat",
#     "numpy",
#     "torch",
#     "torch-einops-utils",
#     "tqdm",
#     "fire",
#     "rich"
# ]
#
# [tool.uv.sources]
# x-neat = { path = "." }
# ///

from __future__ import annotations
import fire
import numpy as np

from rich.console import Console
from rich.table import Table
from rich import box

from neat import NEAT

# environment constants

GRAVITY = 10.0
LENGTH = 1.0
MASS = 1.0
DT = 0.05
MAX_TORQUE = 2.0
MAX_SPEED = 8.0

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

# vectorized multi-condition pendulum environment

class MultiConditionPendulum:
    def __init__(self, pop_size: int, horizon: int = 150, inverted: bool = False):
        self.pop_size = pop_size
        self.horizon = horizon
        self.inverted = inverted
        self.conditions = (
            (-0.5, 0.5),
            (-0.25, -0.5),
            (0.0, 0.0),
            (0.25, 0.5),
            (0.5, -0.5)
        )

    def evaluate(self, pop: NEAT, head: int = 0) -> np.ndarray:
        total_rewards = np.zeros(self.pop_size, dtype = np.float32)

        for th0, thdot0 in self.conditions:
            th = np.full(self.pop_size, th0, dtype = np.float32)
            thdot = np.full(self.pop_size, thdot0, dtype = np.float32)

            for _ in range(self.horizon):
                obs = np.stack((np.cos(th), np.sin(th), thdot), axis = -1)
                u = pop.forward(obs, head = head).squeeze(-1) * MAX_TORQUE
                u = np.clip(u, -MAX_TORQUE, MAX_TORQUE)

                effective_u = -u if self.inverted else u
                costs = wrap_angle(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
                total_rewards -= costs

                accel = (3.0 * GRAVITY / (2.0 * LENGTH) * np.sin(th) + 3.0 / (MASS * LENGTH ** 2) * effective_u)
                new_thdot = thdot + accel * DT
                th = th + new_thdot * DT
                thdot = np.clip(new_thdot, -MAX_SPEED, MAX_SPEED)

        return total_rewards / len(self.conditions)

# continual learning benchmark

def benchmark(
    seeds: tuple[int, ...] = (10, 20, 42, 55, 101),
    pop_size: int = 64,
    t1_target: float = -10.0,
    max_t1_gens: int = 60,
    num_gens_t2: int = 25,
    t2_threshold: float = -50.0,
    connection_cost_penalty: float = 0.05,
    compare_all: bool = True
):
    """
    Continual Learning Benchmark:
    Evaluates downstream transfer to Task 2 (Inverted Dynamics) after pretraining Task 1 (Standard Balance)
    to convergence with Clune connection cost natural selection, compared to standard transfer and from scratch.
    """
    console = Console()

    hparams = dict(
        mutate_prob = 0.9,
        change_edge_weight_prob = 0.6,
        change_node_bias_prob = 0.2,
        grow_node_prob = 0.1,
        grow_edge_prob = 0.1,
        add_novel_edge_prob = 0.2
    )

    env_t1 = MultiConditionPendulum(pop_size, horizon = 150, inverted = False)
    env_t2 = MultiConditionPendulum(pop_size, horizon = 150, inverted = True)

    conditions = [
        ('Transfer (Clune P&CC)', {'pen': connection_cost_penalty}),
    ]
    if compare_all:
        conditions.append(('Transfer (Standard)', {'pen': 0.0}))

    cond_names = [c[0] for c in conditions] + ['From Scratch']
    results = {name: [] for name in cond_names}
    curves = {name: [] for name in cond_names}

    console.rule("[bold magenta]Continual Learning Benchmark (Task 1 Convergence & Frozen Transfer)")

    for seed in seeds:
        console.rule(f"[cyan]Seed {seed}", style = "dim")

        # evaluated transfer conditions

        for name, cfg in conditions:
            np.random.seed(seed)
            net = NEAT(3, 4, 2, pop_size = pop_size, mutation_hyper_params = hparams)

            # train task 1 to convergence with optional clune connection cost pressure

            t1_solved_gen = None
            for g in range(max_t1_gens):
                r_t1 = env_t1.evaluate(net, head = 0)
                best_t1 = float(np.max(r_t1))
                if best_t1 >= t1_target and t1_solved_gen is None:
                    t1_solved_gen = g
                    break

                net.genetic_algorithm_step(
                    r_t1,
                    connection_cost_penalty = cfg['pen']
                )

            eval_t1 = env_t1.evaluate(net, head = 0)
            champ_idx = int(np.argmax(eval_t1))
            net.champion_index = champ_idx
            t1_score = float(eval_t1[champ_idx])
            t1_cost = float(net.connection_costs(squared = True)[0][champ_idx])
            t1_mod = float(net.modularities()[0][champ_idx])
            t1_gens = t1_solved_gen if t1_solved_gen is not None else max_t1_gens

            # continual transfer: seed champion, freeze task 1, train output head 1 on task 2

            net.seed_from_champion()
            net.freeze(except_outputs = 1)

            curve = []
            solved_gen = None
            for g in range(num_gens_t2):
                r_t2 = env_t2.evaluate(net, head = 1)
                best_r = float(np.max(r_t2))
                curve.append(best_r)
                if solved_gen is None and best_r >= t2_threshold:
                    solved_gen = g
                net.genetic_algorithm_step(r_t2)

            # verify zero forgetting on task 1

            t1_post = float(np.max(env_t1.evaluate(net, head = 0)))
            assert abs(t1_post - t1_score) < 1e-4, f"Zero forgetting violated! {t1_post} vs {t1_score}"

            results[name].append({
                'seed': seed, 't1_gens': t1_gens, 't1_score': t1_score,
                't1_cost': t1_cost, 't1_mod': t1_mod,
                'solved_t2': solved_gen, 'final_t2': curve[-1]
            })
            curves[name].append(curve)

            sol_str = f"Gen {solved_gen}" if solved_gen is not None else "Failed"
            console.print(f"  [bold]{name:<24s}[/]: T1 Gens={t1_gens:2d}, Cost={t1_cost:5.2f}, ModQ={t1_mod:5.3f} | T2 Solved={sol_str:>8s}, Final={curve[-1]:8.2f}")

        # from scratch baseline

        np.random.seed(seed)
        net_sc = NEAT(3, 4, 2, pop_size = pop_size, mutation_hyper_params = hparams)
        curve_sc = []
        solved_sc = None
        for g in range(num_gens_t2):
            r_t2 = env_t2.evaluate(net_sc, head = 1)
            best_r = float(np.max(r_t2))
            curve_sc.append(best_r)
            if solved_sc is None and best_r >= t2_threshold:
                solved_sc = g
            net_sc.genetic_algorithm_step(r_t2)

        results['From Scratch'].append({
            'seed': seed, 't1_gens': 0, 't1_score': 0.0, 't1_cost': 0.0, 't1_mod': 0.0,
            'solved_t2': solved_sc, 'final_t2': curve_sc[-1]
        })
        curves['From Scratch'].append(curve_sc)

        sc_sol = f"Gen {solved_sc}" if solved_sc is not None else "Failed"
        console.print(f"  [bold]{'From Scratch':<24s}[/]: T1 Gens= N/A, Cost=  N/A, ModQ=  N/A | T2 Solved={sc_sol:>8s}, Final={curve_sc[-1]:8.2f}")

    # summary table

    console.print()
    table = Table(title = "Continual Learning: Task 1 Convergence & Frozen Transfer", box = box.ROUNDED)
    table.add_column("Condition", style = "bold cyan")
    table.add_column("T1 Gens", justify = "right")
    table.add_column("T1 Cost", justify = "right")
    table.add_column("T1 Mod Q", justify = "right")
    table.add_column("T2 Success", justify = "right", style = "green")
    table.add_column("T2 Solved", justify = "right")
    table.add_column("T2 Final Return", justify = "right", style = "bold yellow")

    for name in cond_names:
        res = results[name]
        succ = [r for r in res if r['solved_t2'] is not None]
        succ_rate = f"{len(succ) / len(res) * 100:.1f}%"
        mean_final = f"{np.mean([r['final_t2'] for r in res]):.2f}"
        mean_solved = f"Gen {np.mean([r['solved_t2'] for r in succ]):.1f}" if succ else "N/A"

        if name != 'From Scratch':
            t1_g = f"{np.mean([r['t1_gens'] for r in res]):.1f}"
            t1_c = f"{np.mean([r['t1_cost'] for r in res]):.2f}"
            t1_m = f"{np.mean([r['t1_mod'] for r in res]):.4f}"
        else:
            t1_g, t1_c, t1_m = "N/A", "N/A", "N/A"

        table.add_row(name, t1_g, t1_c, t1_m, succ_rate, mean_solved, mean_final)

    console.print(table)

    # trajectory table

    console.print()
    traj_table = Table(title = "Task 2 Learning Trajectories (Mean Best Return)", box = box.ROUNDED)
    traj_table.add_column("Gen", justify = "right", style = "dim")
    for name in cond_names:
        traj_table.add_column(name, justify = "right")

    gens_to_show = sorted(set([g for g in [0, 1, 2, 4, 6, 8, 10, 15, 20, num_gens_t2 - 1] if g < num_gens_t2]))
    for g in gens_to_show:
        vals = [f"{np.mean([c[g] for c in curves[name]]):.2f}" for name in cond_names]
        traj_table.add_row(str(g), *vals)

    console.print(traj_table)

if __name__ == '__main__':
    fire.Fire(benchmark)
