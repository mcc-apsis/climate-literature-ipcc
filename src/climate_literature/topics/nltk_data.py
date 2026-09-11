"""One-time NLTK data bootstrap for the topic tokenizer.

The 2020 model's tokenizer used sent_tokenize, POS tagging and WordNet
lemmatisation, each needing its own corpus. On the institute HPC, compute
nodes have no network, so seed these once on a login node with
scripts/seed_nltk_data.py; the default ~/nltk_data lives on the NFS home
mount and compute jobs see it. Afterwards everything is offline and
deterministic.
"""

import nltk

# download name -> nltk.data.find path (the subdirectory differs by type)
PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _find(path: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def _unpack(pkg: str, path: str) -> bool:
    """Unzip a downloaded-but-not-extracted package by hand.

    nltk's downloader can leave <pkg>.zip sitting next to where the
    extracted dir should be (seen with nltk 3.10 on Python 3.14) while
    reporting the package as already up-to-date.
    """
    import zipfile
    from pathlib import Path, PurePosixPath

    dest = Path(nltk.data.path[0]) / PurePosixPath(path).parent
    zipped = dest / f"{pkg}.zip"
    if not zipped.exists():
        return False
    zipfile.ZipFile(zipped).extractall(dest)
    return True


def ensure_nltk_data() -> None:
    """Fetch the corpora the tokenizer needs, verifying each one.

    Verification matters offline: nltk.download() fails silently on an
    unreachable network, and without this check the failure would surface
    mid-tokenisation (or worse, as empty stopwords).
    """
    missing = []
    for pkg, path in PACKAGES.items():
        try:
            nltk.data.find(path)
            continue
        except LookupError:
            nltk.download(pkg, quiet=True)
        try:
            nltk.data.find(path)
        except LookupError:
            if not _unpack(pkg, path) or not _find(path):
                missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"nltk data missing and download failed (offline node?): "
            f"{missing} — seed ~/nltk_data on a login node first."
        )
