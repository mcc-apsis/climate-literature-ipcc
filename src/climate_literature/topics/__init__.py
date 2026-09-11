"""Topic model of the full climate-literature corpus.

Method ported from the NMF topic models of Callaghan et al. (2020),
"A topography of climate change research" — the do_nmf routine of the
tmv/nacsos framework (~/software/nacsos-legacy). Train on a deterministic
sample, apply to the whole corpus. See the module docstrings for the
deliberate deviations from the published method.

The chosen run is released with `finalize` as a re-applyable artifact under
data/topics/models/<tag>/ — pickles plus plain-format copies of the fitted
state plus a golden set — so it can be scored onto a new dataset years later
(see the README.md written there; `verify` checks any environment reproduces
the release before trusting it).

Drive from the repo root, e.g.:

    uv run python -m climate_literature.topics.train sweep --sample 200000
    uv run python -m climate_literature.topics.train finalize --tag K140_a0.1
    uv run python -m climate_literature.topics.train apply --tag K140_a0.1
    uv run python -m climate_literature.topics.train verify --tag K140_a0.1
    uv run python -m climate_literature.topics.compare
"""
