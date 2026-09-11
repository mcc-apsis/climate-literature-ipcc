"""Seed ~/nltk_data with the corpora the topic tokenizer needs.

Run this ONCE on an HPC login node (compute nodes have no internet; the
seeded ~/nltk_data on NFS home is what topics jobs read offline). No repo
sync needed — uv builds a throwaway env:

    uv run --no-project --with nltk scripts/seed_nltk_data.py

Should print "seeded OK". The extraction fallback handles the nltk 3.10 /
py3.14 bug where download() leaves <pkg>.zip unextracted yet reports
"up-to-date" (ensure_nltk_data() in the package has the same fallback).
"""

import zipfile
from pathlib import Path

import nltk

# download name -> nltk.data.find path
PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def found(path: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def main() -> None:
    missing = []
    for pkg, path in PACKAGES.items():
        if found(path):
            continue
        nltk.download(pkg, quiet=True)
        if not found(path):
            # nltk's downloader can leave the zip unextracted and call it done
            zip_path = Path(nltk.data.path[0]) / path.rsplit("/", 1)[0] / f"{pkg}.zip"
            if zip_path.exists():
                zipfile.ZipFile(zip_path).extractall(zip_path.parent)
        if not found(path):
            missing.append(pkg)

    if missing:
        raise SystemExit(f"MISSING after seeding: {missing}")
    print("seeded OK")


if __name__ == "__main__":
    main()
