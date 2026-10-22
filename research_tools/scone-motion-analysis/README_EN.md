# SCONE Motion Analysis — community utility (v3.0 source)

This repository directory is a **maintainer-owned analysis extension** to a fork of [SCONE Core](https://github.com/tgeijten/scone-core). It is not upstream SCONE, and the fork maintainer does not claim to maintain upstream SCONE.

The program is a Python/Tkinter GUI that loads SCONE/OpenSim simulation output (STO, TXT, CSV, XLSX), validates and summarizes time series, calculates selected gait features, and exports tables and visualizations. Source and tests originated from the owner's private `SCONE_analysis_v1` research project. The public release excludes participant data, study results, licensed simulation models and proprietary engine files.

## Quickstart

- Python 3.10+ with Tkinter and pip
- From this directory: `python -m pip install -r requirements.txt`
- Launch: `python main.py`
- Test: `python -m pytest tests -q`

Inputs must be supplied locally by users. Input units and model coordinate signs require independent scientific review. SCONE simulations, licenses and CMA-ES optimization are **not** performed by this GUI.

## Quality & validation

Automated tests are provided, but a copy of tests is not evidence of a successful fresh CI run. Use the dedicated GitHub Actions workflow for the current build, and report reproducible issues with synthetic data.

This project is maintained by [dk-koo](https://github.com/dk-koo). See [CONTRIBUTING.md](CONTRIBUTING.md) and [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md). The fork's Apache-2.0 license covers this directory. For upstream SCONE licensing and attribution, refer to the repository root.
