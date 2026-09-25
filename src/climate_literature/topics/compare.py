"""Topic-comparison sheets for choosing K — the subjective step of 2020.

Two jobs, both driven by the recovered structure of do_nmf's comparison tool
(run_compare_1794_1866.xlsx in the 2019 project):

  ladder sheet — ONE wide sheet per alpha spanning the whole K-series, not a
  sheet per model pair. Columns repeat per run (words | size | similarity to
  next run); rows carry topic identity along the chain. Each topic of a larger
  model is attached to the smaller-model topic it matches most: 1:1 matches
  share the parent's row, splits stack as continuation rows underneath it
  (the parent row holds the best fragment, the rest follow in overlap order).
  So a split reads as a family tree column-wise, and each similarity cell is
  one topic's top-10 word overlap with the parent it hangs from. This replaces
  topic_comparison.xlsx.

  baseline sheet — align a new run's topics against the 2019 published topics
  (recovered topics.csv) by top-word Jaccard, the seed of a later "how has the
  literature changed" comparison.

Sizes are topic term-weight sums from components.parquet — an interim proxy
for the 2019 sheet's aggregate doc-loadings, which only exist per run once it
has gone through `train apply`. Outputs are CSV; no xlsx dependency needed.

Run from the repo root:

    uv run python -m climate_literature.topics.compare ladder --alpha 0.0
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
TOP_N = 10  # words per topic for the similarity metric
DISPLAY_WORDS = 3  # words shown per cell, as in the 2019 sheet

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


def _load_run(run_dir: Path, top_n: int = TOP_N) -> pd.DataFrame:
    """Per-topic top words and term-weight size for one run."""
    comp = pd.read_parquet(run_dir / "components.parquet")
    # cumcount filter: pandas 3.0.5 segfaults on groupby().head()
    ranked = comp.sort_values(["topic", "score"], ascending=[True, False])
    top = ranked[ranked.groupby("topic").cumcount() < top_n]
    words = top.groupby("topic")["term"].apply(list)  # score order, best first
    sizes = comp.groupby("topic")["score"].sum()
    return pd.DataFrame({"words": words, "size": sizes}).reset_index()


def _ladder_rows(
    loaded: dict[int, pd.DataFrame], ks: list[int]
) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    """Genealogy rows: one row per topic-family, splits stack under parents.

    Base run gets one row per topic (sorted by first top word, as in the 2019
    sheet). Each larger run attaches every topic to the smaller-run topic with
    maximal top-word overlap: the best fragment claims the parent's row, the
    rest insert as continuation rows right below it, in overlap order. A parent
    with no children means that topic dissolved (renamed or merged upstream) —
    its later cells read blank.
    """
    rows: list[dict] = []  # each row: {K: topic id} + {f"sim_{ka}-{kb}": int}

    def row_of(kv: int, topic: int) -> dict | None:
        for r in rows:
            if r.get(kv) == topic:
                return r
        return None

    base = loaded[ks[0]].copy()
    base["_first"] = base["words"].str[0]
    for _, t in base.sort_values("_first").iterrows():  # alphabetical, as in 2019
        rows.append({ks[0]: int(t["topic"])})

    stats: dict[tuple[int, int], dict] = {}
    for ka, kb in zip(ks, ks[1:], strict=False):
        words_a = {int(r["topic"]): r["words"] for _, r in loaded[ka].iterrows()}
        # every larger-model topic attaches to its most similar parent topic
        attach: dict[int, list[tuple[int, int]]] = {}  # parent -> [(child, ov)]
        for _, t in loaded[kb].iterrows():
            tb, words_b = int(t["topic"]), t["words"]
            ta, ov = max(
                ((ta, _overlap(words_b, words_a[ta])) for ta in words_a),
                key=lambda x: x[1],
            )
            attach.setdefault(ta, []).append((tb, ov))

        all_sims = []
        for ta, children in attach.items():
            prow = row_of(ka, ta)
            children.sort(key=lambda c: (-c[1], c[0]))
            all_sims.extend(ov for _, ov in children)
            for n, (tb, ov) in enumerate(children):
                entry = {kb: tb, f"sim_K{ka}-K{kb}": ov}
                if n == 0 and prow is not None and kb not in prow:
                    prow.update(entry)
                elif prow is not None:
                    rows.insert(rows.index(prow) + 1 + n, entry)
                else:  # parent had no row of its own: stack after its block
                    rows.append(entry)
        stats[(ka, kb)] = {
            "sims": all_sims,
            # parents with >=2 children; and parents no child attached to
            "splits": sum(len(c) >= 2 for c in attach.values()),
            "dissolved": sum(
                1 for _, r in loaded[ka].iterrows() if int(r["topic"]) not in attach
            ),
        }
    return rows, stats


@app.command()
def ladder(
    k: str = typer.Option("", help="K values in increasing order (default: 80..150)"),
    alpha: float = typer.Option(DEFAULT_ALPHA, help="alpha of the runs to compare"),
) -> None:
    """Write one wide genealogy sheet for a whole K-series at a given alpha."""
    ks = [int(x) for x in k.split(",")] if k else DEFAULT_KS
    loaded: dict[int, pd.DataFrame] = {}
    for kv in ks:
        rd = RUNS_DIR / f"K{kv}_a{alpha}"
        if not (rd / "components.parquet").exists():
            typer.echo(f"skip K={kv}: no run at {rd}")
            continue
        loaded[kv] = _load_run(rd)
    ks_present = [kv for kv in ks if kv in loaded]
    if len(ks_present) < 2:
        raise typer.BadParameter(
            "need at least two comparable runs; run `train sweep` first"
        )

    rows, stats = _ladder_rows(loaded, ks_present)

    def cell(r: dict, col: str):
        if col.startswith("words_"):
            kv = int(col.split("K")[1])
            if kv not in r:
                return ""
            ws = loaded[kv].set_index("topic").loc[r[kv], "words"]
            return "{" + ", ".join(ws[:DISPLAY_WORDS]) + "}"
        if col.startswith("size_"):
            kv = int(col.split("K")[1])
            if kv not in r:
                return ""
            return round(loaded[kv].set_index("topic").loc[r[kv], "size"], 1)
        if col.startswith("sim_"):
            return r.get(col, "")
        return ""

    cols = [
        c
        for i, kv in enumerate(ks_present)
        for c in (
            [f"words_K{kv}", f"size_K{kv}"]
            + ([f"sim_K{kv}-K{ks_present[i + 1]}"] if i + 1 < len(ks_present) else [])
        )
    ]
    out = SHEETS_DIR / f"ladder_a{alpha}.csv"
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{c: cell(r, c) for c in cols} for r in rows]).to_csv(out, index=False)

    for (ka, kb), st in stats.items():
        typer.echo(
            f"K{ka}->K{kb}: mean sim {np.mean(st['sims']):.1f}/{TOP_N}, "
            f"{len(st['sims'])} attached, {st['splits']} splits, "
            f"{st['dissolved']} dissolved"
        )
    typer.echo(f"wrote {out} ({len(rows)} rows)")


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
