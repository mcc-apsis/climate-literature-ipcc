"""Figure builders for the reporting layer.

Each figure is one typer command, e.g.:

    climate-figures papers-by-year

or build them all in one go:

    climate-figures build-all

(Also reachable as `python -m climate_literature.reporting.plots ...`.)
"""

import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import typer
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame, Series, concat

from climate_literature.constants import COORDS_DATA, FIGURES_DIR, PREDICTIONS_DATA
from climate_literature.settings import settings

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


# Corpus figures cover the modern literature only; older records are dropped
# at load time, so every figure shares the same denominator.
MIN_PUBLICATION_YEAR = 1985


def load_predictions(columns: list[str]) -> DataFrame:
    """Read columns from the predictions, with the year cutoff pushed to parquet."""
    return (
        pads.dataset(str(PREDICTIONS_DATA), format="parquet", partitioning="hive")
        .scanner(
            columns=columns,
            filter=pads.field("publication_year") >= MIN_PUBLICATION_YEAR,
        )
        .to_table()
        .to_pandas()
    )


def load_prediction_columns(wanted: list[str]) -> DataFrame:
    """Read chosen columns from the hive-partitioned predictions.

    Partitions from partial runs lack the instrument/sector columns, and
    pyarrow cannot unify the mixed schemas (it takes the schema from the
    first file). So each file contributes the wanted columns it has and
    concat unions the frames, filling absent columns with NaN.

    The publication year is always read when present, to apply the
    MIN_PUBLICATION_YEAR cutoff; rows without year info fall out of it.
    """
    read = [*wanted]
    if "publication_year" not in read:
        read.append("publication_year")
    frames = []
    for path in sorted(PREDICTIONS_DATA.rglob("*.parquet")):
        present = [c for c in read if c in pq.read_schema(path).names]
        frames.append(pq.read_table(path, columns=present).to_pandas())
    df = concat(frames, ignore_index=True)
    df = df[df["publication_year"] >= MIN_PUBLICATION_YEAR]
    if "publication_year" not in wanted:
        df = df.drop(columns="publication_year")
    return df


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
    df = load_predictions(["publication_year"])
    counts = _count_by_year(df["publication_year"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(counts.index, counts.to_numpy(), width=0.8, color=SERIES_1, linewidth=0)
    ax.set_title("Climate change publication growth")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of publications")
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


# Sequential ramp for the embedding panels: one hue, light→dark (magnitude).
# Endpoints are #e8f1fb / SERIES_1 / #14417a — OKLab lightness is strictly
# monotonic across the ramp (0.954 -> 0.378), the checked requirement for a
# sequential map; the categorical adjacency validator would fail by design.
UMAP_RAMP = LinearSegmentedColormap.from_list(
    "umap_blue", ["#e8f1fb", SERIES_1, "#14417a"]
)


def _load_coords() -> DataFrame:
    """UMAP-2D coordinates with item_id normalised to raw 16-byte UUID bytes.

    Both sides of the join arrive as bytes: the arrow uuid extension comes
    back as bytes from to_pandas(), and reduce.py stores the same 16-byte
    binary — so they hash-join directly, no str(UUID) conversion needed.
    """
    path = COORDS_DATA / "coords.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the reduce step "
            "(python -m climate_literature.embed.reduce) or fetch coords.parquet "
            "from the cluster clone's data/coords/."
        )
    return pq.read_table(path, columns=["item_id", "x", "y"]).to_pandas()


@figure
def umap_embedding() -> None:
    """Where the corpus sits in the UMAP-2D embedding: density, time, policy."""
    configure_style()
    coords = _load_coords()
    df = coords.merge(
        load_prediction_columns(["item_id", "relevant", "publication_year"]),
        on="item_id",
        how="inner",
        validate="one_to_one",
    )
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)

    # Shared frame across panels (UMAP axes carry no units, so no ticks).
    padx, pady = (np.ptp(x) * 0.03, np.ptp(y) * 0.03)
    xlim, ylim = (x.min() - padx, x.max() + padx), (y.min() - pady, y.max() + pady)

    fig, (ax_dens, ax_year, ax_pol) = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax in (ax_dens, ax_year, ax_pol):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

    # Panel 1: raw density, with the hero count.
    ax_dens.hexbin(x, y, gridsize=50, cmap=UMAP_RAMP, mincnt=1, linewidths=0)
    ax_dens.text(
        0.02,
        0.98,
        f"{n:,}",
        transform=ax_dens.transAxes,
        fontsize=24,
        color=INK_PRIMARY,
        va="top",
    )
    ax_dens.text(
        0.02,
        0.865,
        "papers embedded",
        transform=ax_dens.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
        va="top",
    )
    ax_dens.set_title("Density", loc="left", fontsize=11, color=INK_SECONDARY)

    # Panels 2-3: per-hex summaries. Bins under mincnt stay unpainted so
    # a stray pair of points cannot mint a whole hex of median/shares.
    hb_year = ax_year.hexbin(
        x,
        y,
        C=df["publication_year"].to_numpy(dtype=float),
        reduce_C_function=np.median,
        gridsize=50,
        mincnt=3,
        cmap=UMAP_RAMP,
        vmin=df["publication_year"].min(),
        vmax=df["publication_year"].max(),
        linewidths=0,
    )
    ax_year.set_title(
        "Median publication year", loc="left", fontsize=11, color=INK_SECONDARY
    )
    cbar_year = fig.colorbar(hb_year, ax=ax_year, fraction=0.046, pad=0.03)
    cbar_year.outline.set_visible(False)
    cbar_year.set_label("year", fontsize=8, color=INK_SECONDARY, labelpad=4)

    # The cascade (classify/predict.py) only scores policy relevance over the
    # corpus, so every joined row carries a score; threshold as elsewhere.
    hb_pol = ax_pol.hexbin(
        x,
        y,
        C=(df["relevant"].to_numpy() > RELEVANCE_THRESHOLD).astype(float),
        reduce_C_function=np.mean,
        gridsize=50,
        mincnt=5,
        cmap=UMAP_RAMP,
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
    )
    ax_pol.set_title(
        "Share on climate policy", loc="left", fontsize=11, color=INK_SECONDARY
    )
    cbar = fig.colorbar(hb_pol, ax=ax_pol, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("share", fontsize=8, color=INK_SECONDARY, labelpad=4)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Strip the frame after drawing (hexbin re-activates axes). Empty tick
    # locators, not axis("off"): the latter hides the parent Axis, leaving
    # the tick-label Text objects themselves visible (and colliding) to any
    # renderer that walks artists by their own visibility flag.
    for ax in (ax_dens, ax_year, ax_pol):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.tight_layout()
    fig.text(
        0.01,
        -0.03,
        f"{settings.embedding_model} embeddings (768-d, L2-normalised) · UMAP 2-D: "
        "n_neighbors=30, min_dist=0, spectral init, fit on a 500k subsample · "
        f"years ≥ {MIN_PUBLICATION_YEAR}",
        fontsize=8,
        color=INK_SECONDARY,
    )
    save_fig(fig, "umap_embedding")
    plt.close(fig)


@app.command()
def build_all() -> None:
    """Rebuild every figure registered with @figure, in registration order."""
    for build in FIGURE_BUILDERS:
        build()


if __name__ == "__main__":
    app()
