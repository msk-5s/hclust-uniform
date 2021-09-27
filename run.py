# SPDX-License-Identifier: MIT

"""
This script runs phase identification using a single set of parameters.

Note that the results in this script may differ a bit from the results gotten from using
`run_suite.py`. This is because the random number generator is only `invoked` once in this script
where as it is `invoked` multiple times in the suite.
"""

import numpy as np
import sklearn.metrics

import model

import data_factory

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
    load = data_factory.make_data(noise_percent=0.005, rng=rng)
    labels = data_factory.make_labels()

    start = 0
    days = 7
    width = 96 * days

    load = load[start:(start + width), :]

    #***********************************************************************************************
    # Run phase identification.
    #***********************************************************************************************
    predictions = model.predict_hclust(labels=labels, load=load, metric="correlation")
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=predictions)

    #***********************************************************************************************
    # Print out results.
    #***********************************************************************************************
    print("-"*50)
    print(f"Accuracy: {accuracy}")
    print("-"*50)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
