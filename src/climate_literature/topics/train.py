"""Fit the NMF topic model on a sample, then apply it to the whole corpus.

Two phases, matching the memory limits of a full-corpus NMF fit:

  sweep  — vectorize a deterministic sample (stratified by publication_year),
           fit NMF for a grid of K/alpha. The fitted vectorizer and model are
           persisted per run under data/topics/runs/<tag>/.

  apply  — load a chosen run's frozen vectorizer + model, transform every
           document in the corpus in shards, landing the doc-topic scores under
           data/topics/doc_topics/<tag>/.

The vectorizer (vocabulary + idf) is fit once on the sample and reused for the
whole corpus — an NMF's components are tied to its vectorizer's vocabulary, so
the same object must produce the training and application matrices. idf is
numerically stable at these scales, so sample idf ≈ full-corpus idf.

NMF settings mirror do_nmf (alpha_W=alpha_H=alpha, l1_ratio=0.1, max_iter=500,
random_state=1). init and solver differ from the published run: init='nndsvda'
tolerates the sample's zero columns, and solver='mu' is far faster than the
default 'cd' on large sparse matrices at the same objective.

Run from the repo root:

    uv run python -m climate_literature.topics.train sweep --sample 200000
    uv run python -m climate_literature.topics.train sweep --limit 20000 --k 80,90
    uv run python -m climate_literature.topics.train apply --tag K140_a0.1
"""

import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import typer
from sklearn.decomposition import NMF

from climate_literature.constants import PREDICTIONS_DATA, TOPICS_DATA
from climate_literature.topics.nltk_data import ensure_nltk_data
from climate_literature.topics.vectorizer import (
    EXTRA_STOPWORDS,
    build_keyword_vocab,
    build_vectorizer,
)

app = typer.Typer(help="Fit and apply the topic model.")

RUNS_DIR = TOPICS_DATA / "runs"
DOC_TOPICS_DIR = TOPICS_DATA / "doc_topics"
MODELS_DIR = TOPICS_DATA / "models"

DEFAULT_KS = [80, 90, 100, 110, 120, 130, 140, 150]
DEFAULT_ALPHAS = [0.01, 0.05, 0.1]
SEED = 1  # the original random_state; also the sample RNG seed


# --- corpus access ---------------------------------------------------------


def _load_corpus(columns: list[str]) -> pd.DataFrame:
    """Read the modeling corpus from the predictions dataset.

    classify/predict.py already translates the raw Scopus jsonl and
    materialises the per-document fields (text, title, publication_year,
    keywords, ...) in data/predictions, dropping records without an abstract
    along the way — so the topic model reads that dataset directly rather
    than re-parsing the raw harvest into a second parquet copy. The one
    corpus-specific transformation is rebuilt here: modeling text = abstract
    + author keywords appended (the keywords also feed the vectorizer's
    phrase vocabulary).
    """
    if not PREDICTIONS_DATA.exists() or not any(PREDICTIONS_DATA.rglob("*.parquet")):
        raise typer.BadParameter(f"no predictions dataset at {PREDICTIONS_DATA}")
    base = {"item_id", "text", "keywords"}
    df = pd.read_parquet(PREDICTIONS_DATA, columns=sorted(base | set(columns)))
    return _modeling_text(df)[columns]


def _modeling_text(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild text = abstract + author keywords appended (no-op without a
    keywords column, so arbitrary external corpora work too)."""
    if "keywords" in df.columns:
        kws = df["keywords"].map(
            lambda k: "; ".join(k) if k is not None and len(k) else ""
        )
        df["text"] = df["text"].astype(str) + "; " + kws
    return df


def _git_commit() -> str:
    """Short hash of the code that produced an artifact (provenance)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic sample with ~proportional representation per year."""
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for _, group in df.groupby("publication_year", dropna=False):
        take = min(int(round(n * len(group) / len(df))), len(group))
        if take > 0:
            idx.extend(rng.choice(group.index.to_numpy(), take, replace=False).tolist())
    sampled = df.loc[idx]
    if len(sampled) > n:  # rounding overflow — trim deterministically
        keep = np.sort(rng.choice(np.arange(len(sampled)), n, replace=False))
        sampled = sampled.iloc[keep]
    return sampled


def _lib_versions() -> dict[str, str]:
    return {
        "sklearn": sklearn.__version__,
        "nltk": nltk.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _components_long(model: NMF, vocab: np.ndarray) -> pd.DataFrame:
    """Topic-term scores as a long (term, topic, score) frame, zero-dropped."""
    wide = pd.DataFrame(model.components_, columns=vocab).T
    wide.index.name = "term"
    long = wide.reset_index().melt(id_vars="term", var_name="topic", value_name="score")
    long["topic"] = long["topic"].astype(int)
    return long[long["score"] > 0]


# --- sweep -----------------------------------------------------------------


@app.command()
def sweep(
    sample: int = typer.Option(200_000, help="Rows to fit each model on"),
    k: str = typer.Option("", help="Comma-separated K values (default: 80..150)"),
    alpha: str = typer.Option("", help="Comma-separated alphas (default: .01,.05,.1)"),
    min_df: int = typer.Option(
        0, help="Min document frequency (0 = auto by sample size)"
    ),
    limit: int = typer.Option(0, help="Cap corpus rows before sampling (smoke tests)"),
) -> None:
    """Fit NMF for a K x alpha grid on a deterministic sample."""
    ensure_nltk_data()
    ks = [int(x) for x in k.split(",")] if k else DEFAULT_KS
    alphas = [float(x) for x in alpha.split(",")] if alpha else DEFAULT_ALPHAS

    df = _load_corpus(["publication_year", "text", "keywords"])
    if limit:
        df = df.iloc[:limit]
    sample_df = _stratified_sample(df, min(sample, len(df)), SEED)
    n_fit = len(sample_df)
    typer.echo(f"fitting on {n_fit:,} sampled rows (of {len(df):,})")

    # Vocabulary density is relative to the FIT set. The original used min_df=50
    # on 400k docs; keep that 50/400k proportion on whatever we fit on, so the
    # sampled model gets a comparable vocabulary (an absolute 150 on a 200k
    # sample would be ~6x too aggressive). Override with an explicit --min-df.
    effective_min_df = min_df or max(5, round(n_fit * 50 / 400_000))
    typer.echo(f"min_df={effective_min_df} (scaled to fit size)")

    texts = sample_df["text"].astype(str).tolist()
    vectorizer = build_vectorizer(
        *build_keyword_vocab(sample_df["keywords"].tolist()), min_df=effective_min_df
    )
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    typer.echo(f"tf-idf matrix: {tfidf.shape}")

    sample_hash = hashlib.sha1(";".join(sorted(texts)).encode()).hexdigest()[:12]

    for kv in ks:
        if kv > tfidf.shape[1]:
            typer.echo(f"skipping K={kv}: more topics than terms")
            continue
        for av in alphas:
            tag = f"K{kv}_a{av}"
            rd = TOPICS_DATA / "runs" / tag
            rd.mkdir(parents=True, exist_ok=True)
            typer.echo(f"fitting NMF {tag} ...")

            model = NMF(
                n_components=kv,
                alpha_W=av,
                alpha_H=av,
                l1_ratio=0.1,
                init="nndsvda",
                solver="mu",
                max_iter=500,
                random_state=SEED,
            )
            model.fit(tfidf)

            # sklearn's W penalty scales with K*n_terms (fixed) while the
            # data term scales with n_docs — so on a small sample or at high K
            # the alpha penalty can crush every component to zero (the original
            # never hit this because it fit on 400k docs). Flag it loudly
            # instead of silently writing an empty, useless run.
            if not model.components_.sum() > 0:
                typer.secho(
                    f"  {tag}: all components zero (over-regularised at this "
                    f"alpha for n_docs={n_fit}, K={kv}) — skipping",
                    fg="yellow",
                )
                continue

            joblib.dump(model, rd / "model.joblib")
            joblib.dump(vectorizer, rd / "vectorizer.joblib")
            pq.write_table(
                pa.Table.from_pandas(_components_long(model, vocab)),
                rd / "components.parquet",
            )
            (rd / "meta.json").write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "K": kv,
                        "alpha": av,
                        "min_df": effective_min_df,
                        "max_df": 0.9,
                        "l1_ratio": 0.1,
                        "init": "nndsvda",
                        "solver": "mu",
                        "max_iter": 500,
                        "random_state": SEED,
                        "n_docs_fit": int(tfidf.shape[0]),
                        "n_terms": int(tfidf.shape[1]),
                        "limit": limit,
                        "extra_stopwords": sorted(EXTRA_STOPWORDS),
                        "libs": _lib_versions(),
                        "sample_hash": sample_hash,
                        "git_commit": _git_commit(),
                    },
                    indent=2,
                )
            )
            typer.echo(f"  wrote {rd}")


# --- apply -----------------------------------------------------------------


@app.command()
def apply(
    tag: str = typer.Option(..., help="Run tag to apply, e.g. K140_a0.1"),
    corpus: str = typer.Option(
        "",
        help="Parquet file/dir with item_id + text (+keywords) to score; "
        "default: data/predictions. Point at a future corpus here to re-apply "
        "the released model to new data.",
    ),
    artifact: str = typer.Option(
        "",
        help="Directory holding model.joblib/vectorizer.joblib; default: the "
        "finalized release under data/topics/models/<tag>, else the sweep run.",
    ),
    shard_rows: int = typer.Option(250_000, help="Rows per output shard"),
) -> None:
    """Score every paper with a saved run, reusing its frozen vectorizer."""
    ensure_nltk_data()
    if artifact:
        rd = Path(artifact)
    elif (MODELS_DIR / tag / "model.joblib").exists():
        rd = MODELS_DIR / tag
    else:
        rd = RUNS_DIR / tag
    if not (rd / "model.joblib").exists():
        raise typer.BadParameter(f"no saved model at {rd}; run `train sweep` first")

    model: NMF = joblib.load(rd / "model.joblib")
    vectorizer = joblib.load(rd / "vectorizer.joblib")
    meta = json.loads((rd / "meta.json").read_text())
    K = meta["K"]

    if corpus:
        avail = pq.ParquetDataset(corpus).schema.names
        missing = {"item_id", "text"} - set(avail)
        if missing:
            raise typer.BadParameter(
                f"corpus {corpus} lacks required column(s): {sorted(missing)}"
            )
        df = _modeling_text(
            pd.read_parquet(
                corpus,
                columns=[c for c in ("item_id", "text", "keywords") if c in avail],
            )
        )
    else:
        df = _load_corpus(["item_id", "text"])
    n = len(df)
    typer.echo(f"transforming {n:,} docs into {K} topics ...")
    out = DOC_TOPICS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    n_shards = int(np.ceil(n / shard_rows))
    for s in range(n_shards):
        lo, hi = s * shard_rows, min((s + 1) * shard_rows, n)
        X = vectorizer.transform(df["text"].iloc[lo:hi].astype(str).tolist())
        H = model.transform(X)
        part = pd.DataFrame(H, columns=[f"topic_{t}" for t in range(K)])
        part.insert(0, "item_id", df["item_id"].iloc[lo:hi].to_numpy())
        part["shard"] = s
        pq.write_to_dataset(
            pa.Table.from_pandas(part),
            root_path=str(out),
            partition_cols=["shard"],
            existing_data_behavior="delete_matching",
        )
        typer.echo(f"  shard {s + 1}/{n_shards} -> {out}")

    (out / "meta.json").write_text(json.dumps({**meta, "n_docs_apply": n}, indent=2))


# --- finalize / verify ------------------------------------------------------


@app.command()
def finalize(
    tag: str = typer.Option(..., help="Run tag to release, e.g. K140_a0.1"),
    n_golden: int = typer.Option(25, help="Docs pinned for future verification"),
) -> None:
    """Assemble a self-contained, long-lived artifact for a chosen run.

    The joblib pickles are the fast path, but they load by reference — they
    need the same class paths and a lenient sklearn. So the fitted state is
    duplicated into plain formats (vocab + idf + keyword vocab parquet/json,
    topic-term weights) that can rebuild the pipeline in a future world where
    the pickles refuse to load, and a golden set pins exact expected scores
    so `verify` can prove any environment (or reconstruction) reproduces the
    2026 model before you trust its output on new data.
    """
    ensure_nltk_data()
    rd = RUNS_DIR / tag
    if not (rd / "model.joblib").exists():
        raise typer.BadParameter(f"no saved model at {rd}; run `train sweep` first")
    model: NMF = joblib.load(rd / "model.joblib")
    vectorizer = joblib.load(rd / "vectorizer.joblib")
    meta = json.loads((rd / "meta.json").read_text())

    md = MODELS_DIR / tag
    md.mkdir(parents=True, exist_ok=True)
    for f in ("model.joblib", "vectorizer.joblib", "components.parquet"):
        shutil.copy2(rd / f, md / f)

    # Plain-format fitted state: TfidfVectorizer vocabulary/idf and the
    # tokenizer's keyword phrase vocab — everything the vectorizer holds
    # beyond sklearn's constructor args.
    vocab = np.asarray(vectorizer.get_feature_names_out())
    pd.DataFrame(
        {"term": vocab, "col": np.arange(len(vocab)), "idf": vectorizer.idf_}
    ).to_parquet(md / "vocab.parquet", index=False)
    tok = vectorizer.tokenizer
    (md / "kw_vocab.json").write_text(
        json.dumps(
            {"kw_text": sorted(tok.kw_text), "kw_ws": sorted(tok.kw_ws)}, indent=2
        )
    )
    comp = pd.read_parquet(rd / "components.parquet")
    # cumcount instead of groupby.head: pandas 3.0.5 segfaults on head().
    ranked = comp.sort_values(["topic", "score"], ascending=[True, False])
    ranked[ranked.groupby("topic").cumcount() < 15].to_parquet(
        md / "topic_words.parquet", index=False
    )

    # Golden set: fixed docs with exact scores, read back by `verify`.
    df = _load_corpus(["item_id", "text"]).sort_values("item_id", kind="stable")
    sel = df.iloc[np.linspace(0, len(df) - 1, min(n_golden, len(df))).astype(int)]
    H = model.transform(vectorizer.transform(sel["text"].astype(str).tolist()))
    golden = sel[["item_id", "text"]].reset_index(drop=True)
    for t in range(H.shape[1]):
        golden[f"score_{t}"] = H[:, t]
    golden.to_parquet(md / "golden.parquet", index=False)

    meta = {
        **meta,
        "finalized": datetime.date.today().isoformat(),
        "git_commit": meta.get("git_commit") or _git_commit(),
        "n_golden": int(len(golden)),
    }
    (md / "meta.json").write_text(json.dumps(meta, indent=2))
    (md / "README.md").write_text(_artifact_readme(tag, meta))
    typer.echo(f"finalized {tag} -> {md}")


def _artifact_readme(tag: str, meta: dict) -> str:
    libs = ", ".join(f"{k}=={v}" for k, v in meta["libs"].items())
    return f"""# Topic model `{tag}` — re-applyable artifact

Assembled {meta["finalized"]} from code at git `{meta["git_commit"]}`, fitted
on {meta["n_docs_fit"]:,} docs / {meta["n_terms"]:,} terms (sample_hash
`{meta["sample_hash"]}`) with: {libs}.
Hyperparameters and vectorizer settings: `meta.json`.

## Re-applying to a new corpus (fast path)

1. Reproduce the environment (login node has internet):

       git checkout {meta["git_commit"] or "the tagged release"}
       uv sync --frozen --extra topics
2. Seed nltk data if `~/nltk_data` is gone: `scripts/seed_nltk_data.py`.
3. Score the new data — a parquet file/dir with `item_id` and `text`
   (+`keywords` if available, appended to the modeling text):

       uv run --frozen --extra topics python -m climate_literature.topics.train \\
           apply --tag {tag} --corpus /path/to/new.parquet

4. Sanity-check the environment first (any version drift shows up here):

       uv run --frozen --extra topics python -m \\
           climate_literature.topics.train verify --tag {tag}

   `golden.parquet` holds {meta.get("n_golden", "?")} pinned docs with exact
   scores; `verify` re-scores them and reports the max deviation.

## If the pickles refuse to load

`model.joblib`/`vectorizer.joblib` unpickle by reference to
`sklearn.decomposition.NMF`, `sklearn.feature_extraction.text.TfidfVectorizer`
and `climate_literature.topics.vectorizer.FancyTokenizer` — do NOT move or
rename that class. If a future sklearn still cannot load them, every fitted
value needed to rebuild the pipeline without refitting is here in plain
formats: `vocab.parquet` (terms + idf), `kw_vocab.json` (phrase merges),
`components.parquet` (topic-term weights). Reconstruct, then reproduce
`golden.parquet` exactly with `verify` before trusting new scores.

## Contents

- `model.joblib` / `vectorizer.joblib` — fitted NMF + TfidfVectorizer (frozen)
- `components.parquet` / `topic_words.parquet` — all term weights / top 15 per topic
- `vocab.parquet`, `kw_vocab.json` — vectorizer state in plain formats
- `golden.parquet` — pinned docs + exact scores (environment check)
- `meta.json` — provenance: hyperparameters, corpus/sample, lib versions, git
"""


@app.command()
def verify(
    tag: str = typer.Option(..., help="Finalized release to check"),
    tol: float = typer.Option(1e-6, help="Max accepted score deviation"),
) -> None:
    """Re-score a finalized release's golden set and compare to pinned scores.

    Run after any environment change: exact agreement means the current
    stack reproduces the released model bit-for-interpretation; deviation
    means do not trust new scores until it is explained.
    """
    ensure_nltk_data()
    md = MODELS_DIR / tag
    for f in ("model.joblib", "vectorizer.joblib", "golden.parquet"):
        if not (md / f).exists():
            raise typer.BadParameter(f"{md / f} missing; run `train finalize` first")
    model: NMF = joblib.load(md / "model.joblib")
    vectorizer = joblib.load(md / "vectorizer.joblib")
    golden = pd.read_parquet(md / "golden.parquet")
    meta = (
        json.loads((md / "meta.json").read_text())
        if (md / "meta.json").exists()
        else {}
    )

    H = model.transform(vectorizer.transform(golden["text"].astype(str).tolist()))
    stored = golden[[f"score_{t}" for t in range(H.shape[1])]].to_numpy()
    max_diff = float(np.abs(H - stored).max())

    now, then = _lib_versions(), meta.get("libs", {})
    if now != then:
        typer.secho(f"env differs from release: {then} -> {now}", fg="yellow")
    if max_diff <= tol:
        typer.secho(
            f"OK: golden scores reproduce (max diff {max_diff:.2e})", fg="green"
        )
    else:
        typer.secho(
            f"FAIL: max diff {max_diff:.2e} exceeds tol {tol:.0e} — do not "
            "trust scores from this environment",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
