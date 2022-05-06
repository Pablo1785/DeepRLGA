from typing import (
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

IND_SIZE = 3
LOW_BOUND = -5.12
UP_BOUND = 5.12
FITNESS_FUNCTION = benchmarks.rastrigin

MATING_RATE = 0.3
INDIVIDUAL_MUTATION_RATE = 0.3
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
    {'function': tools.mutGaussian, 'mu': 0, 'sigma': 1, 'indpb': INDIVIDUAL_MUTATION_RATE},
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
            fitness_fn,
            max_evals: int,
            initial_population_size: int,
            actions_sel: list,
            actions_cx: list,
            actions_mu: list,
            device,
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

        # Possible actions - all combinations of selection, crossover and mutation operators
        self.actions = [(s_idx, c_idx, m_idx) for s_idx in range(len(self.actions_sel)) for c_idx in range(len(
            self.actions_cx))
                        for
                        m_idx
                        in
                        range(len(self.actions_mu))]

        # Episodic variables - these persist only during the episode
        self.current_generation = 0
        self.evals_left = self.max_evals
        self.population = None
        self.done: bool = False

        # Reset episodic variables
        self.reset()

    def take_action(self, action: torch.Tensor):
        """
        :param action: Tensor with a single value - index of chosen action
        :return:
        """
        self.current_state, reward, self.done, _ = self.step(action.item())
        return torch.tensor([reward])

    def register_operators(
            self,
            action_sel_idx: int,
            action_cx_idx: int,
            action_mu_idx: int,
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
        current_record = self.log_stats()
        self.current_state = current_record

        # Check if evolution is finished
        self.done = self.evals_left <= 0

        # Select + Crossover + Mutate
        if not self.done:
            self.evolve()

        return self.current_state, self.get_reward(), self.done, None

    def reset(
            self
            ):
        self._setup_problem()
        self._setup_stats()

        self.current_generation = 0
        self.evals_left = self.max_evals
        self.population = self.toolbox.population(
            n=self.initial_population_size
            )

        self.current_state = {k: 0 for k in self.logbook.header}

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
            )
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
            cxpb=MATING_RATE,
            mutpb=INDIVIDUAL_MUTATION_RATE
            )

        # This produced a new generation
        self.current_generation += 1

        return self.population

    def get_reward(
            self
            ) -> float:
        """
        Return the reward form current state

        :return:
        """
        return self.hof[0].fitness.values[0]

    def _setup_problem(
            self
            ):
        """
        Initialize the current optimization problem

        :param num_dims: Dimensionality of the problme
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

    def _setup_stats(
            self
            ):
        self.hof = tools.HallOfFame(
            maxsize=1,
            similar=lambda
                a,
                b: all(
                a == b
                )
            )

        self.stats = tools.Statistics(
            lambda
                ind: ind.fitness.values
            )
        self.stats.register(
            "avg",
            np.mean
            )
        self.stats.register(
            "std",
            np.std
            )
        self.stats.register(
            "min",
            np.min
            )
        self.stats.register(
            "max",
            np.max
            )

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "avg", "std", "min", "max"

    def get_state(self) -> torch.Tensor:
        if self.done:
            return torch.zeros_like(
                torch.tensor(
                    [0 for k in self.logbook.header]
                    ),
                device=self.device
                ).double()
        else:
            return torch.tensor(
                [self.current_state[k] for k in self.logbook.header],
                device=self.device
                ).double()

    def num_actions_available(self):
        return len(self.actions)

    def get_num_state_features(self):
        return len(self.logbook.header)


def main():
    curr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    em = GeneticAlgorithmEnv(
        num_dims=IND_SIZE,
        low_bound=LOW_BOUND,
        up_bound=UP_BOUND,
        fitness_fn=benchmarks.rastrigin,
        max_evals=MAX_EVALS,
        initial_population_size=INITIAL_POPULATION_SIZE,
        actions_sel=ACTIONS_SEL,
        actions_cx=ACTIONS_CX,
        actions_mu=ACTIONS_MU,
        device=curr_device,
    )

    while not em.done:
        em.step(random.randrange(em.num_actions_available()))
    print(f'Best fitness: {em.get_reward()}')


if __name__ == '__main__':
    main()
