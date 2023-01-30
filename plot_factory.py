# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains a factory for making plots.
"""

from typing import Any, Tuple
from nptyping import NDArray

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_correlation_heatmap(
    data: NDArray[(Any, Any), float], labels: NDArray[(Any,), int]
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a heatmap of the correlations between loads.

    Parameters
    ----------
    data : numpy.ndarray, (n_load, n_timestep)
        The data to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The scree plot figure.
    axs : matplotlib.axes.Axes
        The axis of the scree plot figure.
    """
    # Sort the loads by phase so we can better see how the phases follow the correlation structure
    # of the voltage measurements.
    sort_indices = np.argsort(labels)

    # Boundaries to show the phases more distinctly using horizontal and vertical lines.
    phase_counts = np.bincount(labels)
    boundaries = [phase_counts[0], phase_counts[0] + phase_counts[1]]

    # Tick positions for the phases in the graph axis.
    tick_positions = [
        boundaries[0] // 2,
        boundaries[0] + ((boundaries[1] - boundaries[0]) // 2),
        boundaries[1] + ((len(labels) - boundaries[1]) // 2)
    ]

    tick_labels = ["A", "B", "C"]

    (figure, axs) = plt.subplots()

    cor = np.corrcoef(data[:, sort_indices], rowvar=False)

    image = axs.imshow(cor, aspect="auto")

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("Coefficient")

    figure.tight_layout()
    axs.set_xlabel("Load")
    axs.set_ylabel("Load")
    axs.set_xticks(tick_positions)
    axs.set_yticks(tick_positions)
    axs.set_xticklabels(tick_labels)
    axs.set_yticklabels(tick_labels)

    for boundary in boundaries:
        axs.axhline(y=boundary, color="red", linestyle="dashed")
        axs.axvline(x=boundary, color="red", linestyle="dashed")

    return (figure, axs)
