import time
from typing import (
    Callable,
    List,
)

import numpy as np
from sklearn.cluster import (
    KMeans,
    OPTICS,
)

from scipy.spatial import distance_matrix


def number_of_clusters_diversity(population: list) -> int:
    population_array = np.array(population)

    clustering = OPTICS(min_cluster_size=2).fit(population_array)

    return len(set(clustering.labels_))


def fitness_max_mean_ratio_diversity(population: list) -> float:
    fitness_array = np.array([ind.fitness.values for ind in population])

    if fitness_array.size == 0:
        return 0

    return np.max(fitness_array) / np.mean(fitness_array)


def fitness_mean_min_ratio_diversity(population: list) -> float:
    fitness_array = np.array([ind.fitness.values for ind in population])

    if fitness_array.size == 0:
        return 0

    return np.mean(fitness_array) / np.min(fitness_array)


def gene_mean_std_diversity(population: list) -> float:
    population_array = np.array(population)

    std_per_gene = np.std(population_array, axis=0)

    return np.mean(std_per_gene)


def gene_mean_unique_ratio_diversity(population: list) -> float:
    population_array = np.array(population)

    gene_unique_ratios = np.array([np.unique(gene).size / gene.size for gene in population_array.T])

    return np.mean(gene_unique_ratios)


def clusters_of(fn, n_clusters: int = 4, clustering_method=KMeans, random_seed=0, logbook=None):
    """
    Calculate fn() for each of n_clusters of individuals in population, and return results as numpy array of size
    n_clusters.

    :param fn: Function to calculate for each cluster
    :param population: Population of solutions
    :param n_clusters: Number of clusters
    :param clustering_method:
    :return:
    """
    def _clusterized_fn(population: list) -> np.array:
        population_array = np.array(population)

        model = clustering_method(n_clusters=n_clusters, random_state=random_seed)
        labels = model.fit_predict(population_array)

        cluster_vals = []
        for i in range(n_clusters):
            cluster_vals.append(fn([ind for ind, label in zip(population, labels) if label == i]))
        return np.array(cluster_vals), model.cluster_centers_
    return _clusterized_fn


def get_cluster_mappings(
        gen1_centroids,
        gen2_centroids
        ) -> List[int]:
    """
    Spatially match clusters from one generation to another. This allows to maintain a constant ordering of clusters
    during a single episode.

    :param gen1_centroids:
    :param gen2_centroids:
    :return:
    """
    # Get cluster distances
    c1_c2_dists = distance_matrix(
        gen1_centroids,
        gen2_centroids
        )

    # Get sorted distances
    mappings = []
    for i1 in range(
            c1_c2_dists.shape[0]
            ):
        for i2 in range(
                c1_c2_dists.shape[1]
                ):
            mappings.append(
                (i1, i2, c1_c2_dists[i1, i2])
                )

    mappings_sorted = sorted(
        mappings,
        key=lambda
            m: m[2]
        )

    # Find gen1 to gen2 cluster mappings
    final_mappings = [0] * gen1_centroids.shape[0]
    while mappings_sorted:
        m = mappings_sorted[0]
        final_mappings[m[0]] = m[1]

        new_mappings = []
        for m2 in mappings_sorted:
            if not (m2[0] == m[0] or m2[1] == m[1]):
                new_mappings.append(
                    m2
                    )
        mappings_sorted = new_mappings
    return final_mappings


class Clusterer:
    def __init__(self):
        self.prev_cluster_centers = None
        self.fns = []
        self.n_clusters = 0

    @property
    def prev_cluster_centers_available(self):
        return self.prev_cluster_centers is not None

    def reset(self):
        self.prev_cluster_centers = None

    def clusters_of_fns(
            self,
            fns: List[Callable],
            n_clusters: int = 4,
            clustering_method=KMeans,
            random_seed=0,
            ):
        """
        Calculate each fn() for each of n_clusters of individuals in population, and return results as numpy array of
        size
        n_clusters.

        :param fns: List of functions to calculate for each cluster
        :param population: Population of solutions
        :param n_clusters: Number of clusters
        :param clustering_method:
        :return:
        """
        self.fns = fns
        self.n_clusters = n_clusters

        def _clusterized_fns(
                population: list,
                clusterer = self,
                ) -> np.array:
            population_array = np.array(
                population
                )

            model = clustering_method(
                n_clusters=n_clusters,
                random_state=random_seed
                )
            labels = model.fit_predict(
                population_array
                )

            # Try to match cluster labels to previous clusters
            if clusterer.prev_cluster_centers_available:
                cluster_mappings = get_cluster_mappings(clusterer.prev_cluster_centers, model.cluster_centers_)

                new_labels = np.copy(labels)
                for c1, c2 in enumerate(cluster_mappings):
                    new_labels[labels == c2] = c1
                labels = new_labels

            # Save current cluster centers for later
            clusterer.prev_cluster_centers = model.cluster_centers_

            cluster_vals = []
            for i in range(
                    n_clusters
                    ):
                cluster_population = [ind for ind, label in zip(population, labels) if label == i]
                cluster_vals.append(
                    tuple(fn(cluster_population) for fn in fns)
                    )
            return np.array(
                cluster_vals
                )
        return _clusterized_fns
