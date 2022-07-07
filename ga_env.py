import copy
import json
import time
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import numpy as np
import random

import torch
from deap import base
from deap import creator
from deap import benchmarks
from deap import tools
from deap import algorithms

from deep_rl_ga.diversity import (
    Clusterer,
    fitness_max_mean_ratio_diversity,
    fitness_mean_min_ratio_diversity,
    number_of_clusters_diversity,
    gene_mean_std_diversity,
    gene_mean_unique_ratio_diversity,
)

IND_SIZE = 3
LOW_BOUND = -5.12
UP_BOUND = 5.12
FITNESS_FUNCTION = benchmarks.ackley

ATTRIBUTE_MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 3
INITIAL_POPULATION_SIZE = 150
MAX_EVALS = 10_000

RANDOM_SEED = 0

random.seed(
    RANDOM_SEED
    )
np.random.seed(
    RANDOM_SEED
    )

ACTIONS_SEL = [
    {'function': tools.selTournament, 'tournsize': TOURNAMENT_SIZE},
]

ACTIONS_CX = [
    {'function': tools.cxBlend, 'alpha': UP_BOUND},
]

ACTIONS_MU = [
    {'function': tools.mutGaussian, 'mu': 0, 'sigma': 1, 'indpb': ATTRIBUTE_MUTATION_RATE},
]
ACTIONS_CXPB = [0.05, 0.1, 0.2, 0.4, 0.8]
ACTIONS_MUTPB = [0.05, 0.1, 0.2, 0.4, 0.8]

CLUSTERER = Clusterer()

N_CLUSTERS = 10

STAT_FUNCTIONS = [
    ("clusters_of_multiple_fns", CLUSTERER.clusters_of_fns([
        fitness_max_mean_ratio_diversity,
        fitness_mean_min_ratio_diversity,
        gene_mean_std_diversity,
        gene_mean_unique_ratio_diversity,
        lambda p: len(p) / INITIAL_POPULATION_SIZE,  # Cluster population size as part of initial population size
    ], n_clusters=N_CLUSTERS, random_seed=RANDOM_SEED)),
]

ObsType = TypeVar(
    'ObsType'
    )
ActType = TypeVar(
    'ActType'
    )


class GeneticAlgorithmEnv:
    def __init__(
            self,
            num_dims: int,
            low_bound: float,
            up_bound: float,
            fitness_fn: Callable,
            max_evals: int,
            initial_population_size: int,
            actions_sel: List[Dict],
            actions_cx: List[Dict],
            actions_mu: List[Dict],
            actions_cxpb: List[float],
            actions_mutpb: List[float],
            stat_functions: List[Tuple[str, Callable]],
            clusterer: Clusterer,
            device: torch.device,
            number_of_stacked_states: int = 1,
            optimum_fitness: float = None,
            optimum_fitness_delta: float = 0.0,
    ):
        """

        :param num_dims:
        :param low_bound:
        :param up_bound:
        :param fitness_fn:
        :param max_evals:
        :param initial_population_size:
        :param actions_sel:
        :param actions_cx:
        :param actions_mu:
        :param device: Pytorch device
        """
        self.device = device

        # Problem data
        self.num_dims = num_dims
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.fitness_fn = fitness_fn
        self.max_evals = max_evals
        self.initial_population_size = initial_population_size

        # Action space
        self.actions_sel = actions_sel
        self.actions_cx = actions_cx
        self.actions_mu = actions_mu
        self.actions_cxpb = actions_cxpb
        self.actions_mutpb = actions_mutpb

        # Possible actions - all combinations of selection, crossover and mutation operators
        self.actions = [(s_idx, c_idx, m_idx, cxpb_idx, mutpb_idx) for s_idx in range(len(self.actions_sel)) for c_idx
                        in range(len(
            self.actions_cx))
                        for
                        m_idx
                        in
                        range(len(self.actions_mu)) for cxpb_idx in range(len(self.actions_cxpb)) for mutpb_idx in
                        range(len(self.actions_mutpb))]

        # Episodic variables - these persist only during the episode
        self.current_generation = 0
        self.evals_left = self.max_evals
        self.population = None
        self.done: bool = False

        # Early stopping conditions
        self.optimum_fitness = optimum_fitness
        self.optimum_fitness_delta = optimum_fitness_delta

        # Population data
        self.prev_population = None
        self.prev_fitness = None
        self.prev_fitness_sum = None
        self.curr_fitness_sum = None
        self.prev_best_fitness = None
        self.curr_best_fitness = None

        # Stats
        self.stat_functions = stat_functions
        self.clusterer = clusterer
        self.number_of_stacked_states = number_of_stacked_states

        # Reset episodic variables
        self.reset()

    def to_json(self):
        return json.dumps(
            {
                'num_dims': self.num_dims,
                'low_bound': self.low_bound,
                'up_bound': self.up_bound,
                'fitness_fn': str(self.fitness_fn),
                'max_evals': self.max_evals,
                'initial_population_size': self.initial_population_size,
                'actions_sel': [str(d) for d in self.actions_sel],
                'actions_cx': [str(d) for d in self.actions_cx],
                'actions_mu': [str(d) for d in self.actions_mu],
                'actions_cxpb': [str(d) for d in self.actions_cxpb],
                'actions_mutpb': [str(d) for d in self.actions_mutpb],
                'stat_functions': [str(f) for f in self.stat_functions],
                'clusterer': str(self.clusterer),
                'number_of_stacked_states': self.number_of_stacked_states,
            }
        )

    def take_action(self, action: torch.Tensor):
        """
        :param action: Tensor with a single value - index of chosen action
        :return:
        """
        self.current_state, reward, self.done, self.n_last_states = self.step(action.item())
        return torch.tensor([reward], device=self.device)

    def register_operators(
            self,
            action_sel_idx: int,
            action_cx_idx: int,
            action_mu_idx: int,
            action_cxpb_idx: int,
            action_mutpb_idx: int,
    ):
        """
        Take action by choosing new variational operators

        :param action_sel_idx:
        :param action_cx_idx:
        :param action_mu_idx:
        :return:
        """
        self.toolbox.register(
            "select",
            **self.actions_sel[action_sel_idx % len(
                self.actions_sel
                )]
            )
        self.toolbox.register(
            "mate",
            **self.actions_cx[action_cx_idx % len(
                self.actions_cx
                )]
            )
        self.toolbox.register(
            "mutate",
            **self.actions_mu[action_mu_idx % len(
                self.actions_mu
                )]
            )
        self.cxpb = self.actions_cxpb[action_cxpb_idx]
        self.mutpb = self.actions_mutpb[action_mutpb_idx]

    def step(
            self,
            action_idx: int,
    ) -> Tuple[ObsType, float, bool, dict]:
        # Perform chosen action
        self.register_operators(
            *(self.actions[action_idx])
        )

        # Evaluate
        fits = map(
            self.toolbox.evaluate,
            self.population
            )
        for fit, ind in zip(
                fits,
                self.population
                ):
            ind.fitness.values = fit
        self.evals_left -= len(
            self.population
            )

        # Log data
        self.prev_population = np.array([[gene for gene in ind] for ind in self.population])
        self.prev_fitness = np.array([ind.fitness.values[0] for ind in self.population])

        self.prev_fitness_sum = self.curr_fitness_sum
        self.curr_fitness_sum = np.sum(self.prev_fitness)

        self.prev_best_fitness = self.curr_best_fitness
        self.curr_best_fitness = np.min(self.prev_fitness)

        current_record = self.log_stats()  # performance bottleneck, takes 100% of time for a single NN training step
        modified_record = copy.deepcopy(current_record)
        modified_record.pop('gen')
        modified_record.pop('evals')  # these numbers are too problem specific, ratio of evals left is a better proxy
        # for how much time does the network have left
        self.current_state = modified_record

        self.n_last_states.pop(0)
        self.n_last_states.append(copy.deepcopy(self.current_state))

        # Check if evolution is finished
        self.done = self.evals_left <= 0

        # Check if early stopping condition is met
        if self.optimum_fitness:
            self.done = np.abs(self.curr_best_fitness - self.optimum_fitness) <= self.optimum_fitness_delta

        # Select + Crossover + Mutate
        if not self.done:
            self.evolve()

        return self.current_state, self.get_reward(), self.done, self.n_last_states

    def reset(self):
        self._setup_problem()
        self._setup_stats()

        self.current_generation = 0
        self.evals_left = self.max_evals
        self.population = self.toolbox.population(
            n=self.initial_population_size
            )

        self.current_state = {k: 0 for k in self.logbook.header}
        self.n_last_states: List[Dict] = [
            {k: 0 for k in self.logbook.header} for _ in range(self.number_of_stacked_states)
        ]

    def log_stats(
            self
            ) -> dict:
        """
        Save data about current generation

        :param population:
        :param num_generation:
        :return:
        """
        # Insert best to hall of fame
        self.hof.update(
            self.population
            )

        # Record data
        record = self.stats.compile(
            self.population
            )  # performance bottleneck, takes around 80-90% of time for a single NN training step

        self.logbook.record(
            gen=self.current_generation,
            evals=len(
                self.population
                ),
            **record
            )
        return self.logbook[-1]

    def evolve(
            self
            ):
        # Selection
        chosen_population = self.toolbox.select(
            self.population,
            k=len(
                self.population
                )
            )

        # Apply crossover and mutation
        self.population = algorithms.varAnd(
            chosen_population,
            self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            )

        # This produced a new generation
        self.current_generation += 1

        return self.population

    def get_reward(
            self
            ) -> float:
        """
        Return the reward from current state

        :return:
        """
        reward = 1 / np.abs(self.curr_best_fitness - (self.optimum_fitness if self.optimum_fitness else 0.0))
        if self.done and self.evals_left > 0:
            reward *= (1 + (self.evals_left / self.max_evals)) ** 4  # 99% Unused evals means almost 16x the normal
            # reward
        return reward

    def _setup_problem(
            self
            ):
        """
        Initialize the current optimization problem

        :param num_dims: Dimensionality of the problem
        :param low_bound:
        :param up_bound:
        :param fitness_fn: Function that evaluates the quality of possible solutions
        :return:
        """
        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=(-1.0,)
            )
        creator.create(
            "Individual",
            np.ndarray,
            fitness=creator.FitnessMin
            )

        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "attr_float",
            random.uniform,
            self.low_bound,
            self.up_bound
            )
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=self.num_dims
            )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
            )

        self.toolbox.register(
            "evaluate",
            self.fitness_fn
            )

    def _setup_stats(self):
        self.hof = tools.HallOfFame(
            maxsize=1,
            similar=lambda
                a,
                b: np.all(
                a == b
                )
            )
        self.prev_fitness_sum = np.inf
        self.curr_fitness_sum = np.inf
        self.prev_best_fitness = np.inf
        self.curr_best_fitness = np.inf

        self.stats = tools.Statistics()
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "used_part_of_evals", "left_part_of_evals"

        self.stats.register("used_part_of_evals", lambda _: min(1., 1 - self.evals_left / self.max_evals))
        self.stats.register("left_part_of_evals", lambda _: max(0., self.evals_left / self.max_evals))

        if self.stat_functions:
            for name, fn in self.stat_functions:
                self.stats.register(name, fn)
                self.logbook.header = *self.logbook.header, name

        if self.clusterer:
            self.clusterer.reset()

    def get_state(self) -> torch.Tensor:
        if self.done:
            return torch.zeros(
                    self.get_num_state_features(),
                    device=self.device
                ).double()

        data = []
        for i in reversed(range(self.number_of_stacked_states)):
            for k in self.logbook.header:
                if k in ('gen', 'evals'):
                    continue
                if isinstance(self.n_last_states[i][k], np.ndarray):
                    data += list(self.n_last_states[i][k].flatten())
                else:
                    data.append(self.n_last_states[i][k])
        # Pad data with 0's, e.g. on the first call to get_state() before any state data was collected
        if len(data) < self.get_num_state_features():
            data += [0] * (self.get_num_state_features() - len(data))
        return torch.nan_to_num(torch.tensor(
            data,
            device=self.device
            ).double())

    def num_actions_available(self):
        return len(self.actions)

    def get_num_state_features(self):
        if not self.clusterer.fns:
            return self.number_of_stacked_states * len(self.logbook.header)
        return self.number_of_stacked_states * (len(self.logbook.header) - 3 + len(self.clusterer.fns) *
                                              self.clusterer.n_clusters)  # Clusterer
        # fns + 1 because for each cluster we also calculate its distance from the center of search space
        # logbook.header - 2 because gens and evals are dropped, -1 because clustering data is one of the fields


def main():
    curr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    em = GeneticAlgorithmEnv(
        num_dims=IND_SIZE,
        low_bound=LOW_BOUND,
        up_bound=UP_BOUND,
        fitness_fn=FITNESS_FUNCTION,
        max_evals=MAX_EVALS,
        initial_population_size=INITIAL_POPULATION_SIZE,
        actions_sel=ACTIONS_SEL,
        actions_cx=ACTIONS_CX,
        actions_mu=ACTIONS_MU,
        actions_cxpb=ACTIONS_CXPB,
        actions_mutpb=ACTIONS_MUTPB,
        stat_functions=STAT_FUNCTIONS,
        clusterer=CLUSTERER,
        device=curr_device,
        number_of_stacked_states=4,
    )

    while not em.done:
        em.step(random.randrange(em.num_actions_available()))
        st = em.get_state()
        print(f'\nSTATE: {st}')
    print(f'Best fitness: {1 / em.get_reward()}')


if __name__ == '__main__':
    main()
