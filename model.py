# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for phase label correction.
"""

from typing import Any
from nptyping import NDArray

import numpy as np
import sklearn.cluster
import sklearn.metrics.pairwise

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def predict_hclust(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float], metric: str
) -> NDArray[(Any,), int]:
    """
    Do phase label correction using hierarchical clustering (agglomerative).

    Parameters
    ----------
    load : numpy.ndarray of float, (n_timestep, n_load)
        The load voltages.
    labels : numpy.ndarray of int, (n_load,)
        The true phase labels of each load.
    metric : str, ['correlation', 'euclidean']
        The distance metric to use.
    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted phase labels.
    """
    #***********************************************************************************************
    # Calculate distances.
    #***********************************************************************************************
    dist = None

    if metric == "correlation":
        dist = 1 - np.corrcoef(load, rowvar=False)

    elif metric == "euclidean":
        dist = sklearn.metrics.pairwise.euclidean_distances(load.T)

    else:
        valid_metrics = ["correlation", "euclidean"]
        raise ValueError(f"Invalid metric: '{metric}'. Use {valid_metrics}.")

    #***********************************************************************************************
    # Cluster for identification.
    #***********************************************************************************************
    clusters = sklearn.cluster.AgglomerativeClustering(
        n_clusters=3,
        affinity="precomputed",
        linkage="average",
    ).fit_predict(dist)

    #***********************************************************************************************
    # Phase label correction and accuracy.
    #***********************************************************************************************
    predictions = _predict_majority_vote(clusters=clusters, labels=labels)

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def predict_kmeans(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float], metric: str,
    rng: np.random.Generator) -> NDArray[(Any,), int]:
    """
    Do phase label correction using k-means clustering.

    Parameters
    ----------
    load : numpy.ndarray of float, (n_timestep, n_load)
        The load voltages.
    labels : numpy.ndarray of int, (n_load,)
        The true phase labels of each load.
    metric : str, ['correlation', 'euclidean']
        The distance metric to use.
    rng : Generator
        The random generator to use.
    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted phase labels.
    """
    #***********************************************************************************************
    # Determine data to use.
    #***********************************************************************************************
    data = None

    if metric == "correlation":
        # Directly cluster the correlation matrix.
        data = np.corrcoef(load, rowvar=False)

    elif metric == "euclidean":
        data = load.T

    else:
        valid_metrics = ["correlation", "euclidean"]
        raise ValueError(f"Invalid metric: '{metric}'. Use {valid_metrics}.")

    #***********************************************************************************************
    # Cluster for identification.
    #***********************************************************************************************
    clusters = sklearn.cluster.KMeans(
        n_clusters=3, random_state=rng.integers(9001)
    ).fit_predict(data)

    #***********************************************************************************************
    # Phase label correction and accuracy.
    #***********************************************************************************************
    predictions = _predict_majority_vote(clusters=clusters, labels=labels)

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _predict_majority_vote(
    clusters: NDArray[(Any,), int], labels: NDArray[(Any,), int]
) -> NDArray[(Any,), int]:
    """
    Do label correction using predicted clusters and labels with a majority vote rule.

    Parameters
    ----------
    clusters : numpy.ndarray of int, (n_load,)
        The predicted clusters for each load.
    labels : numpy.ndarray of int, (n_load,)
        The labels to use for the loads. These labels will be used in the majority vote approach.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The label predictions via majority vote.
    """
    unique_clusters = np.unique(clusters)

    indices_list = [np.where(clusters == i)[0] for i in unique_clusters]

    predictions = np.zeros(shape=len(clusters), dtype=int)

    for indices in indices_list:
        observed_labels = labels[indices]
        predicted_label = np.bincount(observed_labels).argmax()

        predictions[indices] = predicted_label

    return predictions
