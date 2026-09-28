from pathlib import Path

RAW_DATA = Path("data/raw/scopus")
MAP_DATA = Path("data/policymap")
PREDICTIONS_DATA = Path("data/predictions")
TOPICS_DATA = Path("data/topics")
EMBEDDINGS_DATA = Path("data/embeddings")
COORDS_DATA = Path("data/coords")
FIGURES_DIR = Path("figures")
# Analysis sheets backing the reports (ladder, baselines, coherence) live in
# git, not in the DVC-tracked data dirs — `dvc checkout` wipes the latter.
TABLES_DIR = Path("report/tables")
