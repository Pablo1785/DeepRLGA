import numpy as np
from sklearn.cluster import OPTICS


def number_of_clusters_diversity(population: list) -> int:
    population_array = np.array(population)

    clustering = OPTICS(min_cluster_size=2).fit(population_array)

    return len(set(clustering.labels_))


def fitness_max_mean_ratio_diversity(population: list) -> float:
    fitness_array = np.array([ind.fitness.values for ind in population])

    return np.max(fitness_array) / np.mean(fitness_array)


def fitness_mean_min_ratio_diversity(population: list) -> float:
    fitness_array = np.array([ind.fitness.values for ind in population])

    return np.mean(fitness_array) / np.min(fitness_array)


def gene_mean_std_diversity(population: list) -> float:
    population_array = np.array(population)

    std_per_gene = np.std(population_array, axis=0)

    return np.mean(std_per_gene)


def gene_mean_unique_ratio_diversity(population: list) -> float:
    population_array = np.array(population)

    gene_unique_ratios = np.array([np.unique(gene).size / gene.size for gene in population_array.T])

    return np.mean(gene_unique_ratios)
