"""Topic-level NPMI coherence over the full corpus, for every run at an alpha.

Builds a binary doc-word presence matrix over the union of all runs' top-N
word lists (all prediction rows, the frozen sweep vectorizer), then scores
each topic as the mean pairwise NPMI of its top words. Mean coherence tells
you the model is healthy; the p10 tail tells you where added topics stop
clarifying and start redistributing — that tail is the K-selection signal.

Vectorisation is checkpointed per corpus shard under data/presence/ (keyed
by sha1 of shard path + word-union hash — growing the union invalidates the
cache), written atomically, so an interrupted scan resumes shard-by-shard.
The cache is derived data: gitignored, regenerable, deliberately outside the
DVC-tracked data/topics/ so `dvc add` never picks it up.

Run from the repo root (heavy: ~30 min on 10 workers for a cold cache):

    uv run --extra topics python -m climate_literature.topics.coherence
    uv run --extra topics python -m climate_literature.topics.coherence --alpha 0.01
"""

import functools
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import typer
from joblib import Parallel, delayed

from climate_literature.constants import PREDICTIONS_DATA, TABLES_DIR
from climate_literature.topics.compare import RUNS_DIR, TOP_N, _load_run
from climate_literature.topics.nltk_data import ensure_nltk_data
from climate_literature.topics.train import _modeling_text

app = typer.Typer(help="NPMI coherence scoring for K selection.")

PRESENCE_DIR = Path("data/presence")


@functools.lru_cache(maxsize=1)
def _get_vec(vec_path: str):
    # loky pickles this by reference; the lru_cache lives per worker process
    ensure_nltk_data()
    return joblib.load(vec_path)


def _presence(path: Path, u_idx: np.ndarray, vec_path: str, out_path: str) -> int:
    """Presence CSR for one corpus shard, checkpointed as npz. Idempotent."""
    if Path(out_path).exists():
        return 0
    names = pq.read_schema(path).names
    df = pd.read_parquet(path, columns=[c for c in ("text", "keywords") if c in names])
    texts = _modeling_text(df)["text"].astype(str).tolist()
    m = _get_vec(vec_path).transform(texts).tocsr()
    m.data = np.ones_like(m.data)
    m = m[:, u_idx]
    tmp = out_path + ".tmp.npz"
    sp.save_npz(tmp, m)
    os.replace(tmp, out_path)
    return 1


def _topic_npmi(
    B: sp.csr_matrix,
    col: np.ndarray,
    n_docs: int,
    words: list[str],
    pos: dict[str, int],
) -> float:
    """Mean pairwise NPMI of one topic's top words over the presence matrix."""
    idx = np.array([pos[w] for w in words])
    S = np.asarray((B[:, idx].T @ B[:, idx]).todense())
    P = S / n_docs
    Pi = col[idx] / n_docs
    eps = 1e-12
    npmi = np.log((P + eps) / (np.outer(Pi, Pi) + eps)) / (-np.log(P + eps))
    iu = np.triu_indices(len(idx), k=1)
    return float(npmi[iu].mean())


@app.command()
def scan(
    alpha: float = typer.Option(0.0, help="score all runs at this alpha"),
    top_n: int = typer.Option(TOP_N, help="words per topic"),
    jobs: int = typer.Option(10, help="vectoriser workers"),
    cache: str = typer.Option(str(PRESENCE_DIR), help="presence checkpoint dir"),
) -> None:
    """Write coherence_a{alpha}[.csv|_detail] to report/tables for all runs."""
    chk_dir = Path(cache)
    tags = sorted(
        rd.name
        for rd in RUNS_DIR.iterdir()
        if (rd / "components.parquet").exists() and rd.name.endswith(f"_a{alpha}")
    )
    if not tags:
        raise typer.BadParameter(f"no runs at alpha={alpha} under {RUNS_DIR}")
    words = {tag: _load_run(RUNS_DIR / tag, top_n) for tag in tags}
    vec_path = str(sorted(RUNS_DIR.glob("K*_a*/vectorizer.joblib"))[0])
    vocab = _get_vec(vec_path).vocabulary_
    union = sorted({w for r in words.values() for ws in r["words"] for w in ws})
    u_idx = np.array([vocab[w] for w in union])
    pos = {w: i for i, w in enumerate(union)}
    typer.echo(f"union of top words: {len(union)} over {len(tags)} runs")

    # checkpoints are column-space-dependent (presence cols = word union), so
    # the union's hash is part of the key — growing the union invalidates them.
    uh = hashlib.sha1("\x1f".join(union).encode()).digest()
    chk_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(PREDICTIONS_DATA.rglob("*.parquet"))
    chk = []
    for f in files:
        key = hashlib.sha1(str(f.resolve()).encode() + b"|" + uh).hexdigest()[:16]
        chk.append(str(chk_dir / f"{key}.npz"))
    todo = sum(1 for c in chk if not Path(c).exists())
    typer.echo(f"{len(files)} corpus files, {todo} to vectorise")
    done = sum(
        Parallel(n_jobs=jobs)(
            delayed(_presence)(f, u_idx, vec_path, c)
            for f, c in zip(files, chk, strict=True)
        )
    )
    typer.echo(f"vectorised {done} new files this run")
    B = sp.vstack([sp.load_npz(c) for c in chk]).tocsc().astype(np.float32)
    n_docs = B.shape[0]
    col = np.asarray(B.sum(axis=0)).ravel()
    typer.echo(f"presence matrix: {B.shape}, {B.nnz / n_docs:.1f} words/doc")

    out, detail = [], []
    for tag in tags:
        r = words[tag]
        tc = np.array(
            [_topic_npmi(B, col, n_docs, t["words"], pos) for _, t in r.iterrows()]
        )
        detail.append(
            pd.DataFrame(
                {"tag": tag, "topic": r["topic"], "npmi": tc, "size": r["size"]}
            )
        )
        out.append(
            {
                "tag": tag,
                "mean_npmi": round(float(tc.mean()), 3),
                "median_npmi": round(float(np.median(tc)), 3),
                "p10_npmi": round(float(np.quantile(tc, 0.1)), 3),
                "n_incoherent(<0)": int((tc < 0).sum()),
            }
        )
    df = pd.DataFrame(out)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"coherence_a{alpha}.csv", index=False)
    pd.concat(detail).to_csv(TABLES_DIR / f"coherence_detail_a{alpha}.csv", index=False)
    typer.echo(df.to_string(index=False))


if __name__ == "__main__":
    # re-import under the canonical name: `-m` binds this file to __main__,
    # and loky workers must resolve the scan targets by module path
    from climate_literature.topics.coherence import app  # ty: ignore[unresolved-import]

    app()
