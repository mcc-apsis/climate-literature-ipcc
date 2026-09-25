"""Fill the WGIII literature-growth paragraph with values from the predictions.

The IPCC skeleton (Section 1, "literature has continued to grow since AR6")
has a bracketed slot for numerical detail. This module recomputes those
numbers from `data/predictions` — the same corpus, loaders and thresholds the
figures use — and renders `templates/wgiii_lit_growth.md.j2` (edit the prose
there) with a values/provenance appendix, so every quoted figure can be
traced back to a filter.

Quote only years up to LAST_COMPLETE_YEAR: the corpus carries in-press cover
dates (records with coverDate 2025-2027 exist), so recent years in the raw
year series are inflated and 2025+ is incomplete.

Run from the repo root:

    uv run python -m climate_literature.reporting.numbers
"""

import math
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from climate_literature.constants import PREDICTIONS_DATA
from climate_literature.reporting.plots import (
    MIN_PUBLICATION_YEAR,
    RELEVANCE_THRESHOLD,
    SECTOR_PREFIX,
    load_prediction_columns,
    sector_columns,
)

app = typer.Typer(help="Compute the numbers quoted in the WGIII text.")

REPORT_DIR = Path("report")
OUT_FILE = REPORT_DIR / "wgiii_lit_growth.md"

# AR6 baseline year (Callaghan et al. 2020 methods re-applied from here) and
# the last year whose cover-date counts are not cut off by the retrieval.
BASELINE_YEAR = 2019
AR6_CUTOFF_FIRST_YEAR = 2022  # AR6 literature searches closed in 2021
LAST_COMPLETE_YEAR = 2024


def _fmt(n: float) -> str:
    return f"{int(round(n)):,}"


def _approx(n: float, sig: int = 2) -> str:
    """Round to `sig` significant figures for prose ('~64,000')."""
    if n == 0:
        return "0"
    mag = 10 ** (int(math.floor(math.log10(abs(n)))) - sig + 1)
    return f"~{round(n / mag) * mag:,.0f}"


def compute_stats() -> dict[str, float]:
    """All quoted values, in one pass over the predictions."""
    df = load_prediction_columns(
        ["item_id", "publication_year", "relevant", *sector_columns()]
    )
    df = df[df["publication_year"] <= LAST_COMPLETE_YEAR]
    modern = df.drop_duplicates("item_id")
    year = modern["publication_year"].astype(int)

    n_corpus = modern["item_id"].nunique()
    counts = year.value_counts().sort_index()
    base, end = counts[BASELINE_YEAR], counts[LAST_COMPLETE_YEAR]
    span = LAST_COMPLETE_YEAR - BASELINE_YEAR
    post_ar6 = int(counts.loc[AR6_CUTOFF_FIRST_YEAR:LAST_COMPLETE_YEAR].sum())

    rel = modern[modern["relevant"] > RELEVANCE_THRESHOLD]
    rel_counts = rel["publication_year"].astype(int).value_counts().sort_index()
    rel_base, rel_end = rel_counts[BASELINE_YEAR], rel_counts[LAST_COMPLETE_YEAR]

    # Sector detail: argmax sector over the recent classified subset, the same
    # argmax rule as reporting.plots.policy_share_by_sector.
    recent_rel = rel[rel["publication_year"] >= AR6_CUTOFF_FIRST_YEAR]
    sectors = recent_rel[[c for c in recent_rel.columns if c.startswith(SECTOR_PREFIX)]]
    sector_counts = sectors.dropna(how="all").idxmax(axis=1).value_counts()
    n_sector = int(sector_counts.sum())

    stats: dict[str, float] = {
        "n_corpus": n_corpus,
        "papers_2019": int(base),
        "papers_2024": int(end),
        "ratio_recent": end / base,
        "cagr_recent": (end / base) ** (1 / span) - 1,
        "papers_post_ar6": post_ar6,
        "share_post_ar6": post_ar6 / n_corpus,
        "n_relevant": int(len(rel)),
        "share_relevant": len(rel) / n_corpus,
        "relevant_2019": int(rel_base),
        "relevant_2024": int(rel_end),
        "cagr_relevant": (rel_end / rel_base) ** (1 / span) - 1,
        "n_sector": n_sector,
    }
    for col, n in sector_counts.items():
        name = col.removeprefix(SECTOR_PREFIX)
        name = name[name.index(". ") + 2 :]
        stats[f"sector_share_{name.lower()}"] = n / n_sector
    return stats


def _sector_str(stats: dict[str, float]) -> str:
    """'energy 31%; …; AFOLU 10%' from the sector_share_* entries, biggest
    first; AFOLU keeps its acronym casing, the rest read lowercase
    mid-sentence."""
    sectors = sorted(
        (
            (
                "AFOLU" if k == "sector_share_afolu" else k[len("sector_share_") :],
                v,
            )
            for k, v in stats.items()
            if k.startswith("sector_share_")
        ),
        key=lambda kv: -kv[1],
    )
    return "; ".join(f"{name} {v:.0%}" for name, v in sectors)


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["comma"] = _fmt
    env.filters["approx"] = _approx
    env.filters["pct"] = lambda x: f"{x:.0%}"
    env.filters["times"] = lambda x: f"{x:.1f}×"
    return env


def render(stats: dict[str, float]) -> str:
    """Fill templates/wgiii_lit_growth.md.j2 — the prose lives there."""
    return (
        _jinja_env()
        .get_template("wgiii_lit_growth.md.j2")
        .render(
            **stats,
            sector_str=_sector_str(stats),
            predictions_dir=str(PREDICTIONS_DATA),
            min_year=MIN_PUBLICATION_YEAR,
            last_year=LAST_COMPLETE_YEAR,
            baseline_year=BASELINE_YEAR,
            ar6_first=AR6_CUTOFF_FIRST_YEAR,
            threshold=RELEVANCE_THRESHOLD,
            sector_prefix=SECTOR_PREFIX,
        )
    )


@app.command()
def fill() -> None:
    """Compute the stats and write the filled WGIII text."""
    stats = compute_stats()
    REPORT_DIR.mkdir(exist_ok=True)
    OUT_FILE.write_text(render(stats))
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    app()
