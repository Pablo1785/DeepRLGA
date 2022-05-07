import copy

from deep_rl_ga.ga_env import GeneticAlgorithmEnv
from typing import (
    Tuple,
    TypeVar,
)
import numpy as np
import random
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from itertools import count
from IPython import display

import torch
from deap import base
from deap import creator
from deap import benchmarks
from deap import tools
from deap import algorithms

from deep_rl_ga.diversity import (
    fitness_max_mean_ratio_diversity,
    fitness_mean_min_ratio_diversity,
    number_of_clusters_diversity,
    gene_mean_std_diversity,
    gene_mean_unique_ratio_diversity,
)


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, env_manager: GeneticAlgorithmEnv):
        self.env_manager = env_manager
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1,
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, fit = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=fit, vmin=self.env_manager.low_bound, vmax=self.env_manager.up_bound,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([self.env_manager.low_bound, self.env_manager.up_bound, self.env_manager.low_bound, self.env_manager.up_bound])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        while not self.env_manager.done:
            self.env_manager.step(random.randrange(self.env_manager.num_actions_available()))

            # Values
            if self.env_manager.prev_population is not None:
                pop_array = copy.deepcopy(self.env_manager.prev_population)
                fit_array = copy.deepcopy(self.env_manager.prev_fitness)

                yield np.c_[pop_array[:, 0], pop_array[:, 1], 1 / fit_array.T]
            else:
                pop_array = np.array(self.env_manager.population)
                fit_array = np.ones(pop_array.shape[0])

                yield np.c_[pop_array[:, 0], pop_array[:, 1], 1 / fit_array.T]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :])
        # Set colors..
        self.scat.set_array(data[:, 2])
        # # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

IND_SIZE = 2
LOW_BOUND = -5.12
UP_BOUND = 5.12
FITNESS_FUNCTION = benchmarks.ackley

MATING_RATE = 0.3
INDIVIDUAL_MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 3
INITIAL_POPULATION_SIZE = 150
MAX_EVALS = 10_000
TOP_BEST_SIZE = 10

RANDOM_SEED = 0

random.seed(
    RANDOM_SEED
    )
np.random.seed(
    RANDOM_SEED
    )

ACTIONS_SEL = [
    {'function': tools.selTournament, 'tournsize': TOURNAMENT_SIZE},
    {'function': tools.selBest, 'k': TOP_BEST_SIZE},
]

ACTIONS_CX = [
    {'function': tools.cxBlend, 'alpha': UP_BOUND},
    {'function': tools.cxTwoPoint},
]

ACTIONS_MU = [
    {'function': tools.mutGaussian, 'mu': 0, 'sigma': 1, 'indpb': INDIVIDUAL_MUTATION_RATE},
    {'function': tools.mutShuffleIndexes, 'indpb': INDIVIDUAL_MUTATION_RATE},
]

if __name__ == '__main__':
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

    a = AnimatedScatter(em)
    plt.show()
