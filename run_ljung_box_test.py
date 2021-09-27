# SPDX-License-Identifier: MIT

"""
This script runs the Ljung-Box test on the load voltage time series.
"""

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools

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
    load = data_factory.make_data(noise_percent=0.0, rng=rng)

    #***********************************************************************************************
    # Run the test for different days.
    #***********************************************************************************************
    start = 0

    # Lags taken from table 1 in Hassani, et al. "Selecting optimal lag order in Ljung-Box test" for
    # a critical value of 0.05.
    hassani_lags = [11, 16, 16, 16, 16, 36, 36]

    frame = pd.DataFrame(
        data = {
            "Days": list(range(1, 8)),
            "Lags": hassani_lags,
            "Absent": [0] * len(hassani_lags),
            "Present": [0] * len(hassani_lags),
            "Percent": [0.0] * len(hassani_lags)
    })

    for (days, lags) in zip(range(1, 8), hassani_lags):
        width = 96 * days
        window = load[start:(start + width), :]

        # The `acf` function can also do the ljung-box test, but the API documentation doesn't match
        # what is actually returned. Instead we seperate out the process.
        acfs = [statsmodels.tsa.stattools.acf(x=x, nlags=lags, fft=True) for x in window.T]
        tests = [statsmodels.tsa.stattools.q_stat(x=acf[1:], nobs=width) for acf in acfs]

        p_values = np.array([test[1][-1] for test in tests])

        critical_value =  0.05
        accept_count = len(np.where(p_values > critical_value)[0])
        reject_count = load.shape[1] - accept_count
        percent = accept_count / load.shape[1]

        frame.loc[days - 1, "Absent"] = accept_count
        frame.loc[days - 1, "Present"] = reject_count
        frame.loc[days - 1, "Percent"] = percent

    #***********************************************************************************************
    # Print out the results.
    #***********************************************************************************************
    print(frame)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
