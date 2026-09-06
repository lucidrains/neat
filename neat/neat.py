from __future__ import annotations

import numpy as np

import nimporter_plus

from neat.neat_nim import (
    add_edge,
    add_node,
    add_topology,
    backprop_nn_single,
    clone_nn,
    crossover_and_add_to_population,
    evaluate_nn_single,
    evaluate_population,
    get_population_complexities,
    get_topology_info,
    init_population as init_population_nim,
    migrate_islands as migrate_nim,
    mutate_all,
    mutate_selected,
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

        # 3. produce the offsprings (brood selection will pick the best of several mutated variants, if brood size > 1)

        use_brood = brood_size > 1 and exists(eval_fn) and len(target_nn_ids) > 0

        if use_brood:
            best_scores = {}
            best_offsprings = {}

            for _ in range(brood_size):
                crossover_and_add_to_population(self.all_top_ids, couples, target_nn_ids, crossover_hyper_params)
                mutate_selected(self.all_top_ids, target_nn_ids, mutation_hyper_params)

                scores = to_score_dict(eval_fn(self, target_nn_ids), target_nn_ids)

                for nn_id in target_nn_ids:
                    score = scores[nn_id]

                    if nn_id not in best_scores or score > best_scores[nn_id]:
                        best_scores[nn_id] = score
                        best_offsprings[nn_id] = [clone_nn(top_id, nn_id) for top_id in self.all_top_ids]

            # 4. commit the best offspring from each brood back into the population

            for nn_id in target_nn_ids:
                for top_id, offspring in zip(self.all_top_ids, best_offsprings[nn_id]):
                    set_nn(top_id, nn_id, offspring)

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
