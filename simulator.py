# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for running different simulation cases.
"""

from typing import Any, Optional
from nptyping import NDArray

import numpy as np
import sklearn.metrics

import model

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_case_1(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float],
    rng: Optional[np.random.Generator] = None # pylint: disable=unused-argument
) -> float:
    """
    Case 1: Phase label correction using hierarchical clustering.

    distance: one minus correlation.

    Parameters
    ----------
    labels : numpy.ndarray of float, (n_load,)
        The true labels.
    load : numpy.ndarray of float, (n_timestep, n_load)
        The window of load data.
    rng : optional of numpy.random.Generator, default=None
        The random generator to use.

    Returns
    -------
    float
        The accuracy.
    """
    predictions = model.predict_hclust(labels=labels, load=load, metric="correlation")
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)

    return accuracy

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_case_2(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float],
    rng: Optional[np.random.Generator] = None # pylint: disable=unused-argument
) -> float:
    """
    Case 2: Phase label correction using hierarchical clustering.

    distance: euclidean.

    Parameters
    ----------
    labels : numpy.ndarray of float, (n_load,)
        The true labels.
    load : numpy.ndarray of float, (n_timestep, n_load)
        The window of load data.
    rng : optional of Generator, default=None
        The random generator to use.

    Returns
    -------
    float
        The accuracy.
    """
    predictions = model.predict_hclust(labels=labels, load=load, metric="euclidean")
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)

    return accuracy

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_case_3(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float],
    rng: Optional[np.random.Generator] = None # pylint: disable=unused-argument
) -> float:
    """
    Case 3: Phase label correction using k-means clustering.

    distance: euclidean distance on correlation matrix.

    Parameters
    ----------
    labels : numpy.ndarray of float, (n_load,)
        The true labels.
    load : numpy.ndarray of float, (n_timestep, n_load)
        The window of load data.
    rng : optional of Generator, default=None
        The random generator to use.

    Returns
    -------
    float
        The accuracy.
    """
    predictions = model.predict_kmeans(labels=labels, load=load, metric="correlation", rng=rng)
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)

    return accuracy

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_case_4(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float],
    rng: Optional[np.random.Generator] = None # pylint: disable=unused-argument
) -> float:
    """
    Case 4: Phase label correction using k-means clustering.

    distance: euclidean distance on the load data.

    Parameters
    ----------
    labels : numpy.ndarray of float, (n_load,)
        The true labels.
    load : numpy.ndarray of float, (n_timestep, n_load)
        The window of load data.
    rng : optional of Generator, default=None
        The random generator to use.

    Returns
    -------
    float
        The accuracy.
    """
    predictions = model.predict_kmeans(labels=labels, load=load, metric="euclidean", rng=rng)
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)

    return accuracy
