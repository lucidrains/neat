from __future__ import annotations

import numpy as np

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
    set_nn
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

class GeneticAlgorithm:
    def stats(self):
        return [get_topology_info(top_id) for top_id in self.all_top_ids]

    def to_json(self):
        return [get_population_json(top_id) for top_id in self.all_top_ids]

    def save_json(self, filename):
        for top_id in self.all_top_ids:
            save_json_to_file(top_id, f'{filename}.id.{top_id}.json')

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
        eps: float = 1e-8,
        brood_size: int = 1,
        child_local_search_size: int = 1,
        prob_structural_brood: float | None = None,
        eval_fn = None
    ):
        # 1. select for fitness, and occasionally simplicity as well

        fitnesses = np.asarray(fitnesses)

        if bernoulli(prob_weigh_complexity_as_fitness):
            complexities = np.array(get_population_complexities(self.all_top_ids[0]))
            simplicity_scores = 1.0 / (complexities + eps)
            fitnesses = fitnesses + simplicity_weight * simplicity_scores

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
        num_recurrent = 0
    ):
        assert len(dims) >= 2

        dim_in = dims[0]
        dim_out = dims[-1]
        dim_hiddens = list(dims[1:-1])

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

    def reset_recurrent_state(self):
        self.recurrent_state = None
        self.single_recurrent_state = None

    def single_forward(
        self,
        index: int,
        state,
        sample = False,
        temperature = 1.
    ):
        has_recurrent = self.num_recurrent > 0

        if has_recurrent:
            self.single_recurrent_state = default(self.single_recurrent_state, np.zeros(self.num_recurrent, dtype = np.float32))
            state = np.concatenate((state, self.single_recurrent_state))

        logits = np.array(evaluate_nn_single(self.top.id, index, state.tolist(), use_exec_cache = True), dtype = np.float32)

        if has_recurrent:
            logits, self.single_recurrent_state = logits[:-self.num_recurrent], logits[-self.num_recurrent:].copy()

        if not sample:
            return logits

        return gumbel_sample(logits, temperature = temperature)

    def forward(
        self,
        state,
        sample = False,
        temperature = 1.,
    ):
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
