"""Reduce stored embeddings to 2-D with UMAP, cached as a coords parquet.

Coordinates are computed once and persisted (never recomputed per figure),
because UMAP determinism is seed-level, not byte-level, and re-fitting is
expensive:

    uv run python -m climate_literature.embed.reduce

Above `fit_subsample` papers, the UMAP is fitted on a seeded random subset
and the full set is projected with `transform` — cheaper and keeps the
global layout stable when the corpus grows. The fitted reducer is saved
alongside so new documents can be projected without re-fitting.

CPU only by default: umap-learn honours random_state deterministically on
CPU (GPU runs introduce float non-determinism across nodes).
"""

import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import umap
from numpy import float32
from numpy.random import default_rng
from pandas import DataFrame
from pyarrow.dataset import dataset

from climate_literature.constants import COORDS_DATA, EMBEDDINGS_DATA


def load_embeddings():
    """(item_ids, X) from the hive-partitioned embeddings dataset."""
    table = dataset(
        str(EMBEDDINGS_DATA), format="parquet", partitioning="hive"
    ).to_table(columns=["item_id", "embedding"])
    n = table.num_rows
    col = table.column("embedding").combine_chunks().flatten()
    dim = len(table.column("embedding")[0].as_py())
    X = col.to_numpy(zero_copy_only=False).astype(float32).reshape(n, dim)
    return table.column("item_id").to_pandas(), X


def main(
    n_neighbors: int = typer.Option(30, help="UMAP neighbourhood size"),
    min_dist: int = typer.Option(0, help="UMAP min point separation"),
    fit_subsample: int = typer.Option(
        500_000, help="Fit on at most this many papers, transform the rest"
    ),
    seed: int = typer.Option(42, help="Random state for fit and subsample"),
):
    """Fit UMAP to 2-D and write data/coords/coords.parquet (+ model)."""
    item_ids, X = load_embeddings()
    n = len(item_ids)
    print(f"{n:,} embeddings, dim {X.shape[1]}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        init="spectral",
        verbose=True,
    )

    if n > fit_subsample:
        # Sorted choice keeps neighbourhood locality checks cheap.
        sample = default_rng(seed).choice(n, size=fit_subsample, replace=False)
        print(f"fitting on {len(sample):,}-paper subsample")
        reducer.fit(X[sample])
        coords = reducer.transform(X)
    else:
        coords = reducer.fit_transform(X)

    COORDS_DATA.mkdir(parents=True, exist_ok=True)
    out = DataFrame({"item_id": item_ids, "x": coords[:, 0], "y": coords[:, 1]})
    pq.write_table(pa.Table.from_pandas(out), COORDS_DATA / "coords.parquet")
    # umap-learn's UMAP has no save() — pickle it (needs joblib at load time).
    joblib.dump(reducer, COORDS_DATA / "umap_model.joblib")
    print(f"wrote {COORDS_DATA / 'coords.parquet'}")


if __name__ == "__main__":
    typer.run(main)
