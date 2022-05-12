import numpy as np
from sklearn.cluster import (
    KMeans,
    OPTICS,
)


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


def clusters_of(fn, n_clusters: int = 4, clustering_model=KMeans):
    """
    Calculate fn() for each of n_clusters of individuals in population, and return results as numpy array of size
    n_clusters.

    :param fn: Function to calculate for each cluster
    :param population: Population of solutions
    :param n_clusters: Number of clusters
    :param clustering_model:
    :return:
    """
    def _clusterized_fn(population: list) -> np.array:
        population_array = np.array(population)

        model = clustering_model(n_clusters=n_clusters)
        labels = model.fit_predict(population_array)

        cluster_vals = []
        for i in range(n_clusters):
            cluster_vals.append(fn([ind for ind, label in zip(population, labels) if label == i]))
        return np.array(cluster_vals), model.cluster_centers_
    return _clusterized_fn
