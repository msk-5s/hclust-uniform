# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for making data.
"""

from typing import Any
from nptyping import NDArray

import numpy as np
import pyarrow.feather

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data(noise_percent: float, rng: np.random.Generator) -> NDArray[(Any, Any), float]:
    """
    Make a window of load data.

    Parameters
    ----------
    noise_percent : float
        The percentage of noise to add to the data.
    rng : numpy.random.Generator
        The random state generator to use.

    Returns
    -------
    load : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    """
    load = pyarrow.feather.read_feather("data/load_voltage.feather").to_numpy(dtype=float)

    # Inject gaussian white noise into the measurements. With the original value as the mean, set
    # the 3-sigma point as the percentage of noise. This ensures that the probability of a noisy
    # sample being within `noise_percent` of the true value is ~99.7% (68/95/99.7 rule).
    load += rng.normal(loc=0, scale=(load * noise_percent) / 3, size=load.shape)

    return load

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_labels() -> NDArray[(Any,), int]:
    """
    Make the labels for the loads.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The phase labels for each load.
    """
    metadata = pyarrow.feather.read_feather("data/metadata.feather")

    phase_labels = metadata["phase"].to_numpy(dtype=int)

    return phase_labels
