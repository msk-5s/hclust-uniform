# SPDX-License-Identifier: MIT

"""
This script runs phase label correction using clustering for different combinations of parameters.
"""

from rich.progress import track

import numpy as np
import pandas as pd

import data_factory
import simulator

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    # We want the results to be repeatable.
    rng = np.random.default_rng(seed=1337)

    labels = data_factory.make_labels()

    #***********************************************************************************************
    # Make simulation parameters.
    #***********************************************************************************************
    # Simulation cases to run.
    cases = {
        "case_1": simulator.run_case_1,
        "case_2": simulator.run_case_2,
        "case_3": simulator.run_case_3,
        "case_4": simulator.run_case_4
    }

    # Window widths to evaluate.
    timesteps_per_day = 96
    widths = [days * timesteps_per_day for days in np.arange(start=1, stop=8, step=1)]

    noise_percents = [0.0, 0.001, 0.002, 0.005]

    # Get indices for the load measurements to keep. This lets us emulate different amounts of
    # smart meter coverage for the network.
    coverage_percents = np.arange(start=0.1, stop=1.1, step=0.1)
    indices_list = []

    for coverage_percent in coverage_percents:
        meter_count = int(labels.size *  coverage_percent)
        indices = rng.permutation(labels.size)[:meter_count]

        indices_list.append(indices)

    # Create an array of [case_name, noise_percent] for all combinations. A data frame of accuracies
    # will be created for each combination.
    combinations = np.array(np.meshgrid(list(cases.keys()), noise_percents)).T.reshape(-1, 2)

    #***********************************************************************************************
    # Run simulation cases.
    #***********************************************************************************************
    for combination in track(combinations, "Processing..."):
        case_name = combination[0]
        noise_percent = float(combination[1])
        run_case = cases[case_name]

        accuracy_df = pd.DataFrame(
            data=0.0, columns=coverage_percents, index=np.array(widths) // timesteps_per_day
        )

        load = data_factory.make_data(noise_percent=noise_percent, rng=rng)

        for (row, width) in enumerate(widths):
            for (col, indices) in enumerate(indices_list):
                accuracy = run_case(labels=labels[indices], load=load[:width, indices], rng=rng)

                accuracy_df.iat[row, col] = accuracy

        # Save the results.
        noise_string = str(round(noise_percent, 3)).replace(".", "p")

        accuracy_df.to_csv(f"results/result-{case_name}-{noise_string}.csv")

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
