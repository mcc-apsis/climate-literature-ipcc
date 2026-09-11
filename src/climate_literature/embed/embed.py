"""Embed papers from the predictions dataset with a sentence-transformer.

Each paper is embedded from title + abstract + author keywords, using the
model named by `embedding_model` in the config (default allenai/sclite-scite,
a distilled SPECTER2 trained for paper title/abstract similarity).

Output is a hive-partitioned parquet dataset (one partition per predictions
source_file) of item_id + float32 embedding, so an interrupted run resumes by
skipping saved partitions — same pattern as classify/predict.py. Sharding for
slurm arrays:

    uv run python -m climate_literature.embed.embed --job-id=$JOB_ID --num-jobs=$N

Embeddings are L2-normalised, so downstream distances are cosine.
"""

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import typer
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer

from climate_literature.constants import EMBEDDINGS_DATA, PREDICTIONS_DATA
from climate_literature.settings import settings


def source_partitions() -> list[str]:
    """Source_file values in the predictions dataset (from dir names only)."""
    return sorted(
        d.name.removeprefix("source_file=")
        for d in PREDICTIONS_DATA.glob("source_file=*")
    )


def saved_source_files() -> set[str]:
    """Source_file values already embedded (partition names, no row reads)."""
    if not EMBEDDINGS_DATA.exists():
        return set()
    return {
        d.name.removeprefix("source_file=")
        for d in EMBEDDINGS_DATA.glob("source_file=*")
    }


def embedding_text(df: DataFrame) -> Series:
    """Title, abstract and keywords joined; empties dropped per paper."""
    title = df["title"].fillna("").astype("string").str.strip()
    abstract = df["text"].fillna("").astype("string").str.strip()
    keywords = df["keywords"].map(
        lambda kws: "; ".join(str(k) for k in kws if k) if kws is not None else ""
    )
    return (
        DataFrame({"t": title, "a": abstract, "k": keywords})
        .apply(lambda r: ". ".join(p for p in (r["t"], r["a"], r["k"]) if p), axis=1)
        .astype(str)
    )


def main(
    job_id: int = typer.Option(0, help="Job ID for distributed processing"),
    num_jobs: int = typer.Option(1, help="Total number of jobs"),
    batch_size: int = typer.Option(64, help="Papers per model.encode batch"),
    max_seq_length: int = typer.Option(512, help="Token cap per paper"),
):
    """Embed papers into vectors, partition by partition, resume-safe."""
    partitions = source_partitions()
    already_saved = saved_source_files()

    model = SentenceTransformer(settings.embedding_model)
    model.max_seq_length = max_seq_length

    for i, source_file in enumerate(partitions):
        if i % num_jobs != job_id:
            continue
        if source_file in already_saved:
            print(f"[yellow]Skipping {source_file} (already embedded)[/yellow]")
            continue

        df = (
            pads.dataset(
                str(PREDICTIONS_DATA / f"source_file={source_file}"),
                format="parquet",
            )
            .to_table(columns=["item_id", "title", "text", "keywords"])
            .to_pandas()
        )
        texts = embedding_text(df)
        if texts.empty:
            continue

        vectors = model.encode(
            texts.tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        out = DataFrame(
            {
                "item_id": df["item_id"],
                "embedding": list(vectors),
                "source_file": source_file,
            }
        )
        pq.write_to_dataset(
            pa.Table.from_pandas(out),
            root_path=str(EMBEDDINGS_DATA),
            partition_cols=["source_file"],
            existing_data_behavior="delete_matching",
        )
        print(f"embedded {len(out)} papers from {source_file}")


if __name__ == "__main__":
    typer.run(main)
