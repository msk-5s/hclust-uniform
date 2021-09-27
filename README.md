# **hclust-uniform**

This repository contains the source code for recreating the research in "Phase Identification of Power Distribution Systems using Hierarchical Clustering Methods". The dataset in this work can be recreated from scratch using the [hclust-uniform-opendss](https://github.com/msk-5s/hclust-uniform-opendss.git) repository.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`data/`
: This folder contains the voltage magnitude dataset, synthetic load profiles, and metadata.

`results/`
: This folder contains the phase identification results for different window widths, noise percents, and meter coverages. These are the results reported in the paper.

## Running
The `run.py` script can be used to run phase identification for a single set of parameters. `run_suite.py` will run phase identification for all different window widths, noise percents, and meter coverages.
