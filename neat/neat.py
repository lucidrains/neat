from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from torch_einops_utils import slice_at_dim, tree_map_tensor

from env_ssl_wrapper import StandardizeEnvWrapper, action_space_is_discrete

import nimporter_plus

from neat.neat_nim import (
    add_edge,
    add_node,
    add_topology,
    backprop_nn_single,
    clone_nn,
    clone_nn_obj,
    crossover_and_add_to_population,
    evaluate_nn_single,
    evaluate_population,
    get_population_complexities,
    get_population_json,
    get_topology_info,
    init_population as init_population_nim,
    migrate_islands as migrate_nim,
    mutate_all,
    mutate_selected_structural,
    mutate_selected_structural_forced,
    mutate_selected_weights,
    mutate_survivors,
    remove_topology,
    reset_top_islands as reset_islands_nim,
    save_json_to_file,
    select_and_tournament,
    set_nn,
    set_frozen,
    set_frozen_subset,
    freeze_population,
    unfreeze_population,
    freeze_subset,
    unfreeze_subset,
    get_frozen_stats,
    get_population_connection_costs,
    get_population_modularity_q
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def log(t, eps = 1e-20):
    return np.log(np.clip(t, a_min = eps, a_max = None))

def bernoulli(p):
    return np.random.binomial(1, p)

def gumbel_sample(logits, temperature = 1., eps = 1e-20):
    if temperature > 0.:
        logits = logits / temperature
        u = np.random.uniform(0., 1., logits.shape).astype(np.float32).clip(eps, 1. - eps)
        logits = logits - log(-log(u, eps), eps)

    return logits.argmax(axis = -1).tolist()

def to_score_dict(scores, target_ids):
    if isinstance(scores, dict):
        return scores

    if len(scores) == len(target_ids):
        return dict(zip(target_ids, scores))

    return {i: scores[i] for i in target_ids}

def get_prob_structural(mutation_hyper_params):
    if not isinstance(mutation_hyper_params, dict):
        return 0.1
    m_prob = mutation_hyper_params.get("mutate_prob", 0.95)
    p_novel = mutation_hyper_params.get("add_novel_edge_prob", 5e-3)
    p_grow_e = mutation_hyper_params.get("grow_edge_prob", 5e-4)
    p_grow_n = mutation_hyper_params.get("grow_node_prob", 1e-5)
    p_toggle = mutation_hyper_params.get("toggle_meta_edge_prob", 0.05)
    p_act = mutation_hyper_params.get("change_activation_prob", 0.001)
    p_node = mutation_hyper_params.get("add_remove_node_prob", 1e-5)

    no_struct = (1.0 - p_novel) * (1.0 - p_grow_e) * (1.0 - p_grow_n) * (1.0 - p_toggle) * (1.0 - p_act) * (1.0 - p_node)
    p_struct = m_prob * (1.0 - no_struct)
    return float(np.clip(p_struct, 0.0, 1.0))


# topology

class Topology:
    def __init__(
        self,
        num_inputs,
        num_outputs,
        pop_size,
        num_hiddens = 32,
        shape: tuple[int, ...] | None = None,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None,
        num_islands = 1,
        num_recurrent = 0
    ):
        if isinstance(num_hiddens, int):
            num_hiddens = (num_hiddens,)

        self.pop_size = pop_size
        self.shape = shape

        if exists(shape):
            assert len(shape) == num_inputs

        self.id = add_topology(
            num_inputs,
            num_outputs,
            num_hiddens,
            mutation_hyper_params,
            crossover_hyper_params,
            selection_hyper_params,
            num_islands,
            num_recurrent
        )

        init_population_nim(self.id, pop_size)

    def init_population(self, pop_size):
        return init_population_nim(self.id, pop_size)

    def add_neuron(self):
        return add_node(self.id)

    def add_synapse(self, from_id, to_id):
        return add_edge(self.id, from_id, to_id)

    def __del__(self):
        try:
            if exists(self.id):
                remove_topology(self.id)
        except Exception:
            pass

# network

class Network:
    def __init__(self, pop, index: int):
        self.pop = pop
        self.index = index

    def forward(self, state, *args, **kwargs):
        if isinstance(state, (list, tuple)):
            state = np.asarray(state, dtype = np.float32)

        return self.pop.single_forward(self.index, state, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset(self):
        self.pop.reset_recurrent_state()

    def __repr__(self):
        return f'Network(index = {self.index})'

# genetic algorithm

class GeneticAlgorithm:
    def __init__(
        self,
        connection_cost_penalty: float = 0.0,
        connection_cost_squared: bool = True
    ):
        self.champion_index = None
        self.connection_cost_penalty = connection_cost_penalty
        self.connection_cost_squared = connection_cost_squared

    def reset_recurrent_state(self):
        pass

    def stats(self):
        return [get_topology_info(top_id) for top_id in self.all_top_ids]

    def to_json(self):
        return [get_population_json(top_id) for top_id in self.all_top_ids]

    def save_json(self, filename):
        for top_id in self.all_top_ids:
            save_json_to_file(top_id, f'{filename}.id.{top_id}.json')

    def __getitem__(self, index: int) -> Network:
        return Network(self, index)

    def freeze(
        self,
        frozen: bool = True,
        node_ids: list[int] | None = None,
        edge_ids: list[int] | None = None,
        except_outputs: int | list[int] | tuple[int, ...] | None = None,
        except_heads: int | list[int] | tuple[int, ...] | None = None
    ):
        except_outputs = default(except_outputs, except_heads)
        except_indices = [except_outputs] if isinstance(except_outputs, int) else list(default(except_outputs, []))

        for top_id in self.all_top_ids:
            if not exists(node_ids) and not exists(edge_ids):
                set_frozen(top_id, frozen, except_indices)
            else:
                set_frozen_subset(top_id, default(node_ids, []), default(edge_ids, []), frozen)
        return self

    def unfreeze(
        self,
        node_ids: list[int] | None = None,
        edge_ids: list[int] | None = None
    ):
        self.freeze(frozen = False, node_ids = node_ids, edge_ids = edge_ids)

    @property
    def frozen_stats(self):
        return [get_frozen_stats(top_id) for top_id in self.all_top_ids]

    @property
    def champion(self) -> Network:
        assert exists(self.champion_index), 'must call genetic_algorithm_step with fitnesses before accessing champion'
        return self[self.champion_index]

    def connection_costs(self, squared: bool = True):
        return [get_population_connection_costs(top_id, squared = squared) for top_id in self.all_top_ids]

    def modularities(self):
        return [get_population_modularity_q(top_id) for top_id in self.all_top_ids]

    def seed_from_champion(self):
        assert exists(self.champion_index), 'must call genetic_algorithm_step with fitnesses before accessing champion'
        for top_id in self.all_top_ids:
            champ_nn = clone_nn(top_id, self.champion_index)
            pop_size = len(self.output) if hasattr(self, 'output') else self.top.pop_size
            for i in range(pop_size):
                set_nn(top_id, i, clone_nn_obj(champ_nn))

    def genetic_algorithm_step(
        self,
        fitnesses,
        selection_hyper_params = None,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        migrate_num = 0,
        reset_islands_num = 0,
        reset_islands_tournament_size = 3,
        prob_weigh_complexity_as_fitness: float = 0.0,
        simplicity_weight: float = 1.0,
        connection_cost_penalty: float | None = None,
        connection_cost_squared: bool | None = None,
        eps: float = 1e-8,
        brood_size: int = 1,
        child_local_search_size: int = 1,
        prob_structural_brood: float | None = None,
        eval_fn = None
    ):
        # 1. select for fitness, and occasionally simplicity as well

        fitnesses = np.asarray(fitnesses)
        self.champion_index = int(np.argmax(fitnesses))

        connection_cost_penalty = default(connection_cost_penalty, getattr(self, 'connection_cost_penalty', 0.0))
        connection_cost_squared = default(connection_cost_squared, getattr(self, 'connection_cost_squared', True))

        if bernoulli(prob_weigh_complexity_as_fitness):
            complexities = np.array(get_population_complexities(self.all_top_ids[0]))
            simplicity_scores = 1.0 / (complexities + eps)
            fitnesses = fitnesses + simplicity_weight * simplicity_scores

        if connection_cost_penalty > 0.0:
            costs = np.array(get_population_connection_costs(self.all_top_ids[0], squared = connection_cost_squared))
            fitnesses = fitnesses - connection_cost_penalty * costs

        # 2. select surviving elites and pairing parents, and determine which offsprings will replace which individuals

        _, _, couples, target_nn_ids = select_and_tournament(self.all_top_ids, fitnesses.tolist(), selection_hyper_params)

        # 3. produce the offsprings
        # supports both Child Local Search (weight optimization on a fixed topology)
        # and Synchronized Structural Broods (macro-level architectural exploration across competing structural hypotheses)

        use_eval = (brood_size > 1 or child_local_search_size > 1) and exists(eval_fn) and len(target_nn_ids) > 0

        if use_eval:
            if brood_size > 1:
                p_struct = prob_structural_brood if exists(prob_structural_brood) else get_prob_structural(mutation_hyper_params)
                is_structural_slot = [bool(bernoulli(p_struct)) for _ in target_nn_ids]
                structural_targets = [tid for tid, is_s in zip(target_nn_ids, is_structural_slot) if is_s]
                structural_couples = [c for c, is_s in zip(couples, is_structural_slot) if is_s]
                parametric_targets = [tid for tid, is_s in zip(target_nn_ids, is_structural_slot) if not is_s]
                parametric_couples = [c for c, is_s in zip(couples, is_structural_slot) if not is_s]
            else:
                structural_targets = []
                structural_couples = []
                parametric_targets = []
                parametric_couples = []

            best_scores = {}
            best_offsprings = {}

            for b in range(brood_size):
                # 3a. Generate candidate topology for this sibling
                if brood_size > 1:
                    if len(structural_targets) > 0:
                        crossover_and_add_to_population(self.all_top_ids, structural_couples, structural_targets, crossover_hyper_params)
                        mutate_selected_structural_forced(self.all_top_ids, structural_targets, mutation_hyper_params)

                    if len(parametric_targets) > 0:
                        crossover_and_add_to_population(self.all_top_ids, parametric_couples, parametric_targets, crossover_hyper_params)
                else:
                    crossover_and_add_to_population(self.all_top_ids, couples, target_nn_ids, crossover_hyper_params)
                    mutate_selected_structural(self.all_top_ids, target_nn_ids, mutation_hyper_params)

                # 3b. Evaluate sibling (with optional child local search over weights)
                if child_local_search_size > 1:
                    base_arch = {
                        tid: [clone_nn(top_id, tid) for top_id in self.all_top_ids]
                        for tid in target_nn_ids
                    }
                    sib_best_scores = {}
                    sib_best_offsprings = {}

                    for k in range(child_local_search_size):
                        for tid in target_nn_ids:
                            for top_id, base_nn in zip(self.all_top_ids, base_arch[tid]):
                                set_nn(top_id, tid, clone_nn_obj(base_nn))

                        mutate_selected_weights(self.all_top_ids, target_nn_ids, mutation_hyper_params)
                        scores = to_score_dict(eval_fn(self, target_nn_ids), target_nn_ids)

                        for tid in target_nn_ids:
                            score = scores[tid]
                            if k == 0 or score > sib_best_scores[tid]:
                                sib_best_scores[tid] = score
                                sib_best_offsprings[tid] = [clone_nn(top_id, tid) for top_id in self.all_top_ids]

                    for tid in target_nn_ids:
                        score = sib_best_scores[tid]
                        if b == 0 or score > best_scores[tid]:
                            best_scores[tid] = score
                            best_offsprings[tid] = sib_best_offsprings[tid]
                else:
                    mutate_selected_weights(self.all_top_ids, target_nn_ids, mutation_hyper_params)
                    scores = to_score_dict(eval_fn(self, target_nn_ids), target_nn_ids)

                    for tid in target_nn_ids:
                        score = scores[tid]
                        if b == 0 or score > best_scores[tid]:
                            best_scores[tid] = score
                            best_offsprings[tid] = [clone_nn(top_id, tid) for top_id in self.all_top_ids]

            # 4. commit the best offspring from each brood back into the population
            for tid in target_nn_ids:
                for top_id, offspring in zip(self.all_top_ids, best_offsprings[tid]):
                    set_nn(top_id, tid, offspring)

            # 5. surviving elites are mutated (offsprings were already mutated as part of brood competition)
            mutate_survivors(self.all_top_ids, mutation_hyper_params)
        else:
            crossover_and_add_to_population(self.all_top_ids, couples, target_nn_ids, crossover_hyper_params)
            mutate_all(self.all_top_ids, mutation_hyper_params)

        # 6. migrate individuals between islands, occasionally resetting the worst islands

        if migrate_num > 0:
            migrate_nim(self.all_top_ids, migrate_num)

        if reset_islands_num > 0:
            reset_islands_nim(self.all_top_ids, fitnesses.tolist(), reset_islands_num, reset_islands_tournament_size)

class NEAT(GeneticAlgorithm):
    def __init__(
        self,
        *dims,
        pop_size,
        mutation_hyper_params = None,
        crossover_hyper_params = None,
        selection_hyper_params = None,
        num_islands = 1,
        num_recurrent = 0,
        connection_cost_penalty: float = 0.0,
        connection_cost_squared: bool = True
    ):
        assert len(dims) >= 2
        super().__init__(
            connection_cost_penalty = connection_cost_penalty,
            connection_cost_squared = connection_cost_squared
        )

        dim_in = dims[0]
        dim_out = dims[-1]
        dim_hiddens = list(dims[1:-1])

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_recurrent = num_recurrent
        self.output = np.empty((pop_size, dim_out + num_recurrent), dtype = np.float32)
        self.recurrent_state = None
        self.single_recurrent_state = None

        self.top = Topology(
            dim_in,
            dim_out,
            num_hiddens = dim_hiddens,
            pop_size = pop_size,
            mutation_hyper_params = mutation_hyper_params,
            crossover_hyper_params = crossover_hyper_params,
            selection_hyper_params = selection_hyper_params,
            num_islands = num_islands,
            num_recurrent = num_recurrent
        )

        self.all_top_ids = [self.top.id]

    @property
    def router(self):
        if not hasattr(self, '_router'):
            self._router = TaskRouter(self)
        return self._router

    def register_task(self, *args, **kwargs):
        return self.router.register(*args, **kwargs)

    def freeze_task(self, *args, **kwargs):
        return self.router.freeze(*args, **kwargs)

    def reset_recurrent_state(self):
        self.recurrent_state = None
        self.single_recurrent_state = None

    def single_forward(
        self,
        index: int,
        state,
        sample = False,
        temperature = 1.,
        head: int | slice | tuple[int, int] | None = None,
        task: str | tuple[object, object] | None = None
    ):
        if exists(task):
            return self.router.single_forward(index, state, task, sample = sample, temperature = temperature)

        is_torch = torch.is_tensor(state)
        state = tree_map_tensor(lambda t: t.detach().cpu().numpy(), state)
        state = np.asarray(state, dtype = np.float32)

        has_recurrent = self.num_recurrent > 0

        if has_recurrent:
            self.single_recurrent_state = default(self.single_recurrent_state, np.zeros(self.num_recurrent, dtype = np.float32))
            state = np.concatenate((state, self.single_recurrent_state))

        logits = np.array(evaluate_nn_single(self.top.id, index, state.tolist(), use_exec_cache = True), dtype = np.float32)

        if has_recurrent:
            logits, self.single_recurrent_state = logits[:-self.num_recurrent], logits[-self.num_recurrent:].copy()

        if exists(head):
            head = slice(*head) if isinstance(head, tuple) else (slice(head, head + 1) if isinstance(head, int) else head)
            logits = slice_at_dim(logits, head, dim = -1)

        if is_torch:
            logits = torch.from_numpy(logits)

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

    def forward(
        self,
        state,
        sample = False,
        temperature = 1.,
        head: int | slice | tuple[int, int] | None = None,
        task: str | tuple[object, object] | None = None,
    ):
        if exists(task):
            return self.router.forward(state, task, sample = sample, temperature = temperature)

        is_torch = torch.is_tensor(state)
        state = tree_map_tensor(lambda t: t.detach().cpu().numpy(), state)
        state = np.asarray(state, dtype = np.float32)

        has_recurrent = self.num_recurrent > 0

        if has_recurrent:
            batch, *_ = state.shape
            self.recurrent_state = default(self.recurrent_state, np.zeros((batch, self.num_recurrent), dtype = np.float32))
            state = np.concatenate((state, self.recurrent_state), axis = -1)

        input = np.ascontiguousarray(state, dtype = np.float32)

        evaluate_population(self.top.id, input, self.output)

        out = self.output

        if has_recurrent:
            out, self.recurrent_state = out[:, :-self.num_recurrent], out[:, -self.num_recurrent:].copy()

        if exists(head):
            head = slice(*head) if isinstance(head, tuple) else (slice(head, head + 1) if isinstance(head, int) else head)
            out = slice_at_dim(out, head, dim = -1)

        if is_torch:
            out = torch.from_numpy(out)

        if not sample:
            return out

        return gumbel_sample(out, temperature = temperature)

    def backprop(
        self,
        index: int,
        state,
        target,
        learning_rate: float = 0.01
    ):
        backprop_nn_single(
            self.top.id,
            index,
            state.tolist(),
            target.tolist(),
            learning_rate
        )

# task routing

class TaskRouter:
    """
    Manages task routing and head selection across distinct tasks.
    """
    def __init__(self, neat: NEAT):
        self.neat = neat
        self.tasks = {}

    def register(
        self,
        name: str,
        output_head: int | slice | tuple[int, int] | None = None,
        input_head: int | slice | tuple[int, int] | None = None,
        head: int | slice | tuple[int, int] | None = None
    ):
        self.tasks[name] = {
            'input_head': input_head,
            'output_head': default(output_head, head)
        }
        return self

    def embed_input(self, state, input_head):
        if not exists(input_head):
            return state

        is_torch = torch.is_tensor(state)
        state_np = tree_map_tensor(lambda t: t.detach().cpu().numpy(), state) if is_torch else state
        state_np = np.asarray(state_np, dtype = np.float32)

        slc = slice(input_head, input_head + 1) if isinstance(input_head, int) else (slice(*input_head) if isinstance(input_head, tuple) else input_head)
        full_state = np.zeros((*state_np.shape[:-1], self.neat.dim_in), dtype = np.float32)
        full_state[..., slc] = state_np

        return torch.from_numpy(full_state) if is_torch else full_state

    def forward(self, state, task: str | tuple[object, object], *args, **kwargs):
        if isinstance(task, tuple) and len(task) == 2:
            input_head, output_head = task
        else:
            assert task in self.tasks, f'Task {task} not found'
            task_cfg = self.tasks[task]
            input_head, output_head = task_cfg['input_head'], task_cfg['output_head']

        state = self.embed_input(state, input_head)
        return self.neat.forward(state, *args, head = output_head, **kwargs)

    def single_forward(self, index: int, state, task: str | tuple[object, object], *args, **kwargs):
        if isinstance(task, tuple) and len(task) == 2:
            input_head, output_head = task
        else:
            assert task in self.tasks, f'Task {task} not found'
            task_cfg = self.tasks[task]
            input_head, output_head = task_cfg['input_head'], task_cfg['output_head']

        state = self.embed_input(state, input_head)
        return self.neat.single_forward(index, state, *args, head = output_head, **kwargs)

    def freeze(self, task: str, except_other_tasks: list[str] | None = None):
        assert task in self.tasks, f'Task {task} not found'
        except_outputs = []
        if exists(except_other_tasks):
            for other in except_other_tasks:
                oh = self.tasks[other]['output_head']
                if exists(oh):
                    if isinstance(oh, int):
                        except_outputs.append(oh)
                    elif isinstance(oh, (tuple, list)):
                        except_outputs.extend(oh)
                    elif isinstance(oh, slice):
                        except_outputs.extend(range(oh.start or 0, oh.stop or self.neat.dim_out, oh.step or 1))
        self.neat.freeze(except_outputs = except_outputs)
        return self

# behavior cloning

def behavior_clone(
    env,
    champion,
    mlp,
    save_path: str | Path = 'mlp.pt',
    *,
    iterations: int = 20,
    rollouts_per_iter: int = 4,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    student_action_prob: float = 0.2,
    max_steps: int = 1000,
    target_ratio: float = 0.95,
    min_reward: float | None = None,
    discrete: bool | None = None,
    eval_every: int = 5,
    eval_episodes: int = 5,
    eval_seed: int = 42,
    device: str = 'cpu',
    show_progress: bool = True,
    verbose: bool = True
):
    # standardize environment

    if not isinstance(env, StandardizeEnvWrapper):
        env = StandardizeEnvWrapper(env, device = device)

    # resolve teacher policy callable and reset

    if isinstance(champion, GeneticAlgorithm):
        champion = champion.champion

    if isinstance(champion, Network):
        teacher_policy = champion
        teacher_reset = champion.reset
    elif isinstance(champion, tuple) and len(champion) == 2:
        pop, idx = champion
        teacher_policy = lambda obs: pop.single_forward(idx, obs, sample = False)
        teacher_reset = pop.reset_recurrent_state
    elif callable(champion):
        teacher_policy = champion
        teacher_reset = None
    else:
        raise ValueError(f'Unsupported champion type: {type(champion)}')

    # determine if action space is discrete or continuous

    discrete = default(discrete, action_space_is_discrete(env.action_space))

    device = torch.device(device)
    mlp = mlp.to(device)
    optimizer = Adam(mlp.parameters(), lr = lr)

    def action_from_teacher(t_out):
        if discrete:
            return int(np.argmax(t_out)) if (isinstance(t_out, np.ndarray) and t_out.size > 1) else int(t_out)
        return torch.from_numpy(np.asarray(t_out, dtype = np.float32)).to(device)

    def student_act(obs):
        with torch.no_grad():
            out = mlp(obs).squeeze(0)

        if discrete:
            return int(out.argmax())
        return out

    # loss function

    loss_fn = F.cross_entropy if discrete else F.mse_loss

    def evaluate(policy, reset_fn = None, num_episodes = eval_episodes, start_seed = eval_seed):
        rewards = []

        for episode in range(num_episodes):
            if callable(reset_fn):
                reset_fn()

            obs, _ = env.reset(seed = start_seed + episode)
            episode_reward = 0.0

            for _ in range(max_steps):
                action = policy(obs)
                obs, reward, *_ = env.step(action)
                episode_reward += float(reward.sum().item())

                if env.all_done:
                    break

            rewards.append(episode_reward)

        return float(np.mean(rewards)), float(np.std(rewards))

    # evaluate teacher baseline

    teacher_policy_fn = lambda obs: action_from_teacher(teacher_policy(obs.squeeze(0).cpu().numpy()))
    teacher_reward, teacher_std = evaluate(teacher_policy_fn, reset_fn = teacher_reset)

    threshold = default(
        min_reward,
        teacher_reward * target_ratio if teacher_reward >= 0. else teacher_reward * (1. - target_ratio)
    )

    if verbose:
        print(f'teacher reward: {teacher_reward:.2f} ± {teacher_std:.2f} | target: {threshold:.2f}')

    # DAgger behavior cloning loop

    pbar = range(iterations)
    if show_progress:
        pbar = tqdm(pbar, desc = 'behavior cloning')

    all_states = []
    all_targets = []

    for it in pbar:
        for _ in range(rollouts_per_iter):
            if callable(teacher_reset):
                teacher_reset()

            obs, _ = env.reset()

            for _ in range(max_steps):
                teacher_output = teacher_policy(obs.squeeze(0).cpu().numpy())
                teacher_action = action_from_teacher(teacher_output)

                all_states.append(obs.squeeze(0).detach().cpu())
                all_targets.append(teacher_action)

                action = student_act(obs) if (it > 0 and np.random.rand() < student_action_prob) else teacher_action

                obs, *_ = env.step(action)
                if env.all_done:
                    break

        if len(all_states) == 0:
            continue

        states_t = torch.stack(all_states).to(device)

        if discrete:
            targets_t = torch.tensor(all_targets, dtype = torch.long, device = device)
        else:
            targets_t = torch.stack([torch.as_tensor(t, device = device) for t in all_targets])

        # train

        num_samples = states_t.shape[0]
        curr_batch_size = min(batch_size, num_samples)

        mlp.train()
        for _ in range(epochs):
            perm = torch.randperm(num_samples)
            for start in range(0, num_samples, curr_batch_size):
                idx = perm[start:start + curr_batch_size]
                loss = loss_fn(mlp(states_t[idx]), targets_t[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # check performance against environment

        if divisible_by(it + 1, eval_every) or it == iterations - 1:
            mlp.eval()
            student_reward, _ = evaluate(student_act)

            if student_reward >= threshold:
                if verbose:
                    print(f'reached target: mlp reward {student_reward:.2f} >= {threshold:.2f}')
                break

    # save

    save_path = Path(save_path)
    save_path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(mlp.state_dict(), str(save_path))

    if verbose:
        print(f'saved cloned mlp to {save_path}')

    return mlp
