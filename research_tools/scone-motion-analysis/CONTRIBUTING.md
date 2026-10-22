# Contributing to SCONE Motion Analysis

Research reproducibility and data protection are the priorities.

1. Report issues with Python version, OS, steps to reproduce and a **synthetic** minimal input. Never post human participant records, proprietary models, confidential manuscripts, paths containing personal information or license keys.
2. Open a pull request for a focused change with tests. Run `python -m pytest tests -q` and describe affected units, time handling and biomechanical assumptions.
3. Preserve source values and report conversions explicitly. Do not silently interpolate or fabricate missing data. Treat any scientific validity claim as requiring independent verification.
4. The maintainer reviews issue triage, fixes and releases. Contributions are considered under Apache-2.0 (the repository root license).

Feature directions: independent CLI/batch analysis; explicit STO/radian-degree rules; unit-aware joint load reporting; gait-event detection validation; deterministic provenance summaries; SCONE/OpenSim compatibility tests.
