import json
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from nacsos_data.util.academic.apis.scopus import ScopusAPI
from rich import print
from transformers import pipeline

from climate_literature.constants import RAW_DATA
from climate_literature.settings import settings

out_dir = Path("data/predictions")

KEEP_COLUMNS = [
    "item_id",
    "text",
    "title",
    "publication_year",
    "source",
    "keywords",
    "authors",
    "scopus_id",
]


def predict(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if (
        model_name != settings.inclusion_model
        and settings.inclusion_label in df.columns
    ):
        df = df[df[settings.inclusion_label] > 0.5]

    if df.empty:
        return df[["item_id"]].copy()

    texts = df["text"].astype(str).tolist()
    item_ids = df["item_id"]
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    results = cast(
        list[list[dict[str, float]]],
        classifier(texts, truncation=True, max_length=512),
    )
    row_dict = [{item["label"]: item["score"] for item in row} for row in results]
    pred_df = pd.DataFrame(row_dict)
    pred_df["item_id"] = item_ids.values
    return pred_df


def read_scopus_into_df(jsonl_file: Path) -> pd.DataFrame:
    with jsonl_file.open() as f:
        records = [
            ScopusAPI.translate_record(json.loads(line)).model_dump() for line in f
        ]
        df = pd.DataFrame(records)[KEEP_COLUMNS]
        return df


def main(
    job_id: int = typer.Option(0, help="Job ID for distributed processing"),
    num_jobs: int = typer.Option(1, help="Total number of jobs"),
):
    """Make predictions on the raw data and turn into a parquet dataset."""

    for i, jsonl_file in enumerate(sorted(RAW_DATA.glob("*.jsonl"))):
        if i % num_jobs != job_id:
            continue
        batch_df = read_scopus_into_df(jsonl_file).head(10)

        batch_df = batch_df[batch_df["text"].str.contains(r"\w")]
        for model in settings.models:
            batch_df = batch_df.merge(predict(batch_df, model), how="left")
            print(batch_df)

        batch_df["source_file"] = jsonl_file.name

        table = pa.Table.from_pandas(batch_df)
        pq.write_to_dataset(
            table,
            root_path=str(out_dir),
            partition_cols=["source_file"],
            existing_data_behavior="delete_matching",
        )


if __name__ == "__main__":
    typer.run(main)
