"""Figure builders for the reporting layer.

Each figure is one typer command, e.g.:

    uv run python -m climate_literature.reporting.plots papers-by-year

or build them all in one go:

    uv run python -m climate_literature.reporting.plots build-all
"""

import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import typer
from pandas import DataFrame, Series, concat

from climate_literature.constants import FIGURES_DIR, PREDICTIONS_DATA

app = typer.Typer(help="Build the reporting figures.")


@app.callback()
def main() -> None:
    """Figure builders (subcommands keep working as commands are added)."""


# Roster of figure builders, populated by the @figure decorator below.
FIGURE_BUILDERS: list[Callable[[], None]] = []


def figure(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a builder as both a subcommand and a build-all member."""
    FIGURE_BUILDERS.append(fn)
    return app.command()(fn)


# Ink and grid tokens (recessive axes, dark-on-light text).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e4e1"

# Categorical slot 1 — reuse in order across figures, never cycle.
SERIES_1 = "#2a78d6"

# Recessive neutral for "everything else" context segments — deliberately
# achromatic (not a series), validated against SERIES_1 for CVD separation.
NEUTRAL_REST = "#9d9c98"


def configure_style() -> None:
    plt.rcParams.update(
        {
            # Committed SVGs must be byte-deterministic: the default hasalt is a
            # per-process uuid4, so clip-path/glyph ids churn on every run.
            "svg.hashsalt": "climate-literature",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.linewidth": 0.8,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.svg"
    # metadata Date=None drops the embedded timestamp (with svg.hashsalt this
    # makes unchanged data produce byte-identical files).
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    print(f"wrote {path}")


def load_predictions() -> pads.Dataset:  # noqa: ANN401
    return pads.dataset(str(PREDICTIONS_DATA), format="parquet", partitioning="hive")


def load_prediction_columns(wanted: list[str]) -> DataFrame:
    """Read chosen columns from the hive-partitioned predictions.

    Partitions from partial runs lack the instrument/sector columns, and
    pyarrow cannot unify the mixed schemas (it takes the schema from the
    first file). So each file contributes the wanted columns it has and
    concat unions the frames, filling absent columns with NaN.
    """
    frames = []
    for path in sorted(PREDICTIONS_DATA.rglob("*.parquet")):
        present = [c for c in wanted if c in pq.read_schema(path).names]
        frames.append(pq.read_table(path, columns=present).to_pandas())
    return concat(frames, ignore_index=True)


# The sector model's columns look like "8 - 04. Energy" in the predictions.
SECTOR_PREFIX = "8 - "
RELEVANCE_THRESHOLD = 0.5  # matches the cascade in classify/predict.py


def sector_columns() -> list[str]:
    """Sector-score column names present anywhere in the predictions."""
    return sorted(
        {
            c
            for path in PREDICTIONS_DATA.rglob("*.parquet")
            for c in pq.read_schema(path).names
            if c.startswith(SECTOR_PREFIX)
        }
    )


def _count_by_year(column: Series) -> Series:
    """Per-year counts, dropping records without a publication year."""
    return column.dropna().astype(int).value_counts().sort_index()


@figure
def papers_by_year() -> None:
    """Total number of papers in the corpus by publication year."""
    configure_style()
    df = load_predictions().to_table(columns=["publication_year"]).to_pandas()
    counts = _count_by_year(df["publication_year"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(counts.index, counts.to_numpy(), width=0.8, color=SERIES_1, linewidth=0)
    ax.set_title("Papers in the corpus by publication year")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Papers")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "papers_by_year")
    plt.close(fig)


@figure
def policy_share_by_sector() -> None:
    """Share of the corpus on climate policy, and those papers by sector."""
    configure_style()
    sectors = sector_columns()
    df = load_prediction_columns(["relevant", *sectors])

    n_total = len(df)
    n_policy = int((df["relevant"] > RELEVANCE_THRESHOLD).sum())
    share = n_policy / n_total

    # The sector model only ran on papers above the relevance threshold
    # (the cascade in classify/predict.py), so per-sector totals are the
    # argmax sector over classified papers.
    classified = df[sectors].dropna()
    counts = classified.idxmax(axis=1).value_counts()
    counts = counts.reindex(sectors, fill_value=0).sort_values()
    labels = [
        re.sub(r"^\d+\.\s*", "", c.removeprefix(SECTOR_PREFIX)) for c in counts.index
    ]
    vmax = max(int(counts.max()), 1)

    fig, (ax_share, ax_sector) = plt.subplots(
        1,
        2,
        figsize=(9.5, 3.4),
        width_ratios=[1, 1.4],
    )

    # Left: one part-to-whole bar; the headline share as a hero number.
    gap = 0.008  # surface gap between the two segments
    ax_share.barh(
        [0],
        [1.0 - share - gap],
        left=share + gap,
        height=0.55,
        color=NEUTRAL_REST,
        linewidth=0,
    )
    ax_share.barh([0], [share], height=0.55, color=SERIES_1, linewidth=0)
    ax_share.text(0, 0.38, f"{share:.1%}", fontsize=24, color=INK_PRIMARY, va="bottom")
    ax_share.text(
        0,
        0.27,
        f"{n_policy:,} of {n_total:,} papers",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    # Segment key below the bar (text wears ink; the markers carry colour).
    # Label offset = marker half-width (~0.021 data units at markersize 7)
    # plus a visible air gap.
    ax_share.plot(
        [0.004],
        [-0.5],
        marker="s",
        markersize=7,
        ls="none",
        color=SERIES_1,
        clip_on=False,
    )
    ax_share.text(
        0.034, -0.5, "on climate policy", fontsize=9, color=INK_SECONDARY, va="center"
    )
    ax_share.plot(
        [0.42],
        [-0.5],
        marker="s",
        markersize=7,
        ls="none",
        color=NEUTRAL_REST,
        clip_on=False,
    )
    ax_share.text(
        0.454, -0.5, "rest of corpus", fontsize=9, color=INK_SECONDARY, va="center"
    )
    ax_share.set_xlim(0, 1)
    ax_share.set_ylim(-0.68, 0.66)
    ax_share.axis("off")
    ax_share.set_title(
        "Share of the corpus on climate policy",
        loc="left",
        fontsize=11,
        color=INK_SECONDARY,
    )

    # Right: per-sector totals, sorted, direct-labelled (no x axis needed).
    ax_sector.barh(
        range(len(counts)), counts.to_numpy(), height=0.62, color=SERIES_1, linewidth=0
    )
    for i, v in enumerate(counts.to_numpy()):
        ax_sector.text(
            v + vmax * 0.02, i, f"{v:,}", va="center", fontsize=9, color=INK_SECONDARY
        )
    ax_sector.set_yticks(range(len(counts)), labels)
    ax_sector.set_xlim(0, vmax * 1.15)
    ax_sector.set_xticks([])
    ax_sector.spines[["top", "right", "bottom"]].set_visible(False)
    ax_sector.set_title(
        "Policy papers by sector", loc="left", fontsize=11, color=INK_SECONDARY
    )
    ax_sector.text(
        0,
        -0.08,
        f"n = {len(classified):,} sector-classified policy papers",
        transform=ax_sector.transAxes,
        fontsize=8,
        color=INK_SECONDARY,
    )

    # wspace is applied after tight_layout, not via gridspec_kw: a
    # locally-modified gridspec makes the axes report as incompatible with
    # tight_layout, which then warns and silently skips them.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    save_fig(fig, "policy_share_by_sector")
    plt.close(fig)


@app.command()
def build_all() -> None:
    """Rebuild every figure registered with @figure, in registration order."""
    for build in FIGURE_BUILDERS:
        build()


if __name__ == "__main__":
    app()
