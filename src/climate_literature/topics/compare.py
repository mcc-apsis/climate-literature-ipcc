"""Topic-comparison sheets for choosing K — the subjective step of 2020.

Two jobs, both driven by the recovered structure of do_nmf's comparison tool:

  adjacent-K sheet — for each neighbouring pair in a K sweep, align topics by
  top-word overlap (the original's top_word_overlap method) and lay them out
  side by side, so a human can see how the topic set fragments as K grows and
  pick a granularity. This replaces topic_comparison.xlsx.

  baseline sheet — align a new run's topics against the 2019 published topics
  (recovered topics.csv) by top-word Jaccard, the seed of a later "how has the
  literature changed" comparison.

Outputs are CSV (Excel-openable); no xlsx dependency needed.

Run from the repo root:

    uv run python -m climate_literature.topics.compare adjacent --alpha 0.1
    uv run python -m climate_literature.topics.compare baseline --tag K140_a0.1
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.optimize import linear_sum_assignment

from climate_literature.constants import TOPICS_DATA

app = typer.Typer(help="Compare topic-model runs for K selection.")

RUNS_DIR = TOPICS_DATA / "runs"
SHEETS_DIR = TOPICS_DATA / "comparison"

DEFAULT_KS = [80, 90, 100, 110, 120, 130, 140, 150]
DEFAULT_ALPHA = 0.1
TOP_N = 10

RECOVERED_2019 = Path(
    "~/Documents/papers/published/cc-topography-recovered/tables/topics.csv"
)


# --- loading top words -----------------------------------------------------


def _top_words_by_topic(run_dir: Path, top_n: int = TOP_N) -> dict[int, list[str]]:
    """topic id -> its top_n words, from a saved run's components.parquet."""
    comp = pd.read_parquet(run_dir / "components.parquet")
    out: dict[int, list[str]] = {}
    for topic, group in comp.groupby("topic"):
        out[int(str(topic))] = group.nlargest(top_n, "score")["term"].tolist()
    return out


def _topic_ids_words(
    csv_path: Path, words_col: str, id_col: str
) -> dict[int, list[str]]:
    """topic id -> words for the recovered 2019 sheet (top_words is a list repr)."""
    df = pd.read_csv(csv_path.expanduser())
    return {
        int(row[id_col]): ast.literal_eval(row[words_col]) for _, row in df.iterrows()
    }


# --- comparison cores ------------------------------------------------------


def _overlap(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def _jaccard(a: list[str], b: list[str]) -> float:
    union = set(a) | set(b)
    return len(set(a) & set(b)) / len(union) if union else 0.0


def _align(
    words_a: dict[int, list[str]], words_b: dict[int, list[str]], sim
) -> pd.DataFrame:
    """Globally align two topic sets (scipy linear_sum_assignment on similarity).

    Produces one row per topic in B, with its best 1:1-matched topic from A
    (empty if A ran out). Handles unequal K via padding columns.
    """
    ids_a, ids_b = list(words_a), list(words_b)
    if not ids_a or not ids_b:
        return pd.DataFrame()

    size = max(len(ids_a), len(ids_b))
    score = np.zeros((size, size))
    for i, ta in enumerate(ids_a):
        for j, tb in enumerate(ids_b):
            score[i, j] = sim(words_a[ta], words_b[tb])

    row_ind, col_ind = linear_sum_assignment(-score)  # maximise similarity

    rows = []
    matched = set()
    for i, j in zip(row_ind, col_ind, strict=True):
        if j < len(ids_b) and i < len(ids_a):
            rows.append(
                {
                    "topic_b": ids_b[j],
                    "words_b": words_b[ids_b[j]],
                    "topic_a": ids_a[i],
                    "words_a": words_a[ids_a[i]],
                    "overlap": _overlap(words_b[ids_b[j]], words_a[ids_a[i]]),
                    "similarity": round(float(score[i, j]), 3),
                }
            )
            matched.add(j)
    # Unmatched B topics (when B larger than A) get an empty pairing.
    for j, tb in enumerate(ids_b):
        if j not in matched:
            rows.append(
                {
                    "topic_b": tb,
                    "words_b": words_b[tb],
                    "topic_a": None,
                    "words_a": [],
                    "overlap": 0,
                    "similarity": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values("topic_b").reset_index(drop=True)


def _flatten_words(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        words_b=df["words_b"].map(" ".join),
        words_a=df["words_a"].map(lambda w: " ".join(w) if isinstance(w, list) else ""),
    )


# --- commands --------------------------------------------------------------


@app.command()
def adjacent(
    k: str = typer.Option("", help="K values compared in order (default: 80..150)"),
    alpha: float = typer.Option(DEFAULT_ALPHA, help="alpha of the runs to compare"),
    top_n: int = typer.Option(TOP_N, help="Top words per topic"),
) -> None:
    """Write side-by-side topic sheets for each neighbouring pair in a K sweep."""
    ks = [int(x) for x in k.split(",")] if k else DEFAULT_KS
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    loaded: dict[int, dict] = {}
    for kv in ks:
        rd = RUNS_DIR / f"K{kv}_a{alpha}"
        if not (rd / "components.parquet").exists():
            typer.echo(f"skip K={kv}: no run at {rd}")
            continue
        loaded[kv] = _top_words_by_topic(rd, top_n)

    ks_present = [kv for kv in ks if kv in loaded]
    if len(ks_present) < 2:
        raise typer.BadParameter(
            "need at least two comparable runs; run `train sweep` first"
        )

    for ka, kb in zip(ks_present, ks_present[1:], strict=False):
        aligned = _align(loaded[ka], loaded[kb], _overlap)
        out = SHEETS_DIR / f"compare_K{ka}_vs_K{kb}_a{alpha}.csv"
        _flatten_words(aligned).to_csv(out, index=False)
        mean_ov = aligned["overlap"].mean()
        typer.echo(f"K{ka}->K{kb}: mean overlap {mean_ov:.1f}/{top_n}, wrote {out}")


@app.command()
def baseline(
    tag: str = typer.Option(..., help="New run tag, e.g. K140_a0.1"),
    reference: str = typer.Option(str(RECOVERED_2019), help="2019 topics.csv"),
    top_n: int = typer.Option(TOP_N, help="Top words per topic"),
) -> None:
    """Align a new run against the recovered 2019 topics by top-word Jaccard."""
    rd = RUNS_DIR / tag
    if not (rd / "components.parquet").exists():
        raise typer.BadParameter(f"no run at {rd}; run `train sweep` first")
    ref_path = Path(reference)
    if not ref_path.expanduser().exists():
        raise typer.BadParameter(f"no reference topics at {ref_path}")

    new_words = _top_words_by_topic(rd, top_n)
    ref_words = _topic_ids_words(ref_path.expanduser(), "top_words", "topic_id")

    aligned = _align(ref_words, new_words, _jaccard)
    out = SHEETS_DIR / f"baseline_{tag}_vs_2019.csv"
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    _flatten_words(aligned).to_csv(out, index=False)

    matched = (aligned["overlap"] > 0).mean()
    typer.echo(
        f"{len(new_words)} new topics vs {len(ref_words)} 2019 topics: "
        f"{matched:.0%} share at least one top word. wrote {out}"
    )


if __name__ == "__main__":
    app()
