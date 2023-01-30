# SPDX-License-Identifier: BSD-3-Clause

"""
This script makes a heatmap plot of a correlation matrix of the load voltages.
"""

import matplotlib.pyplot as plt
import numpy as np

import data_factory
import plot_factory

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    # We want and randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    load = data_factory.make_data(noise_percent=0.0, rng=rng)
    labels = data_factory.make_labels()

    start = 0
    days = 7
    width = 96 * days

    load = load[start:(start + width), :]

    #***********************************************************************************************
    # Set font sizes.
    #***********************************************************************************************
    #fontsize = 40
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    #***********************************************************************************************
    # Plot the correlation heatmap.
    #***********************************************************************************************
    _ = plot_factory.make_correlation_heatmap(data=load, labels=labels)

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
