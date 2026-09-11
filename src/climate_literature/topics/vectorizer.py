"""Tokenizer + vectorizer for the topic model.

A port of the `fancy_tokenize` tokenizer inside the tmv/nacsos do_nmf routine
(~/software/nacsos-legacy/BasicBrowser/tmv_app/tasks.py). It does two things a
plain lemmatising tokenizer does not:

1. Keyword-phrase merging. Author keywords that are multi-word or hyphenated
   and occur in a reasonable number of documents (corpus-size // 200 here, as
   in the original) are treated as single tokens: a matched phrase is merged to
   one hyphenated token and removed from the text before the lemmatising pass,
   so "carbon capture" becomes the token "carbon-capture" rather than two
   loose lemmas.

2. POS-aware WordNet lemmatisation (wordpunct -> pos_tag -> lemmatize), with
   the same token filters as the original: drop stopwords, all-punctuation,
   length < 3, all-digits.

Extra stopwords added to the tokenizer (from the original RunStats): the
publisher boilerplate that leaked into 2019-era abstracts.
"""

import re
from collections import Counter
from collections.abc import Iterable

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

EXTRA_STOPWORDS = {
    "elsevier",
    "rights",
    "reserved",
    "john",
    "wiley",
    "sons",
    "copyright",
}

# POS tag -> WordNet part-of-speech for the lemmatizer (original mapping).
_TAG_MAP = {"N": wn.NOUN, "V": wn.VERB, "R": wn.ADV, "J": wn.ADJ}

# Keyword phrases must contain a separator (space or hyphen) to be merged.
_MULTIWORD = re.compile(r"\w+[\-\ ]")

_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def build_keyword_vocab(keywords_by_doc: list[list[str]]) -> tuple[set, set]:
    """Return (keyword_phrases, keyword_first_words) to seed phrase merging.

    keywords_by_doc is one author-keyword sequence per modeling document
    (a list on the way in, a numpy array once round-tripped through parquet,
    so we duck-type rather than checking for list). Only multi-word/hyphenated
    phrases appearing in more than n_docs//200 documents are merged — the
    original's threshold, which keeps genuinely topical phrases and drops
    long-tail one-offs.
    """
    n_docs = max(len(keywords_by_doc), 1)
    counts: Counter = Counter()
    for kws in keywords_by_doc:
        if kws is None or isinstance(kws, str) or not len(kws):
            continue
        seen = {
            kw.replace("-", " ").strip().lower()
            for kw in kws
            if isinstance(kw, str) and _MULTIWORD.search(kw)
        }
        counts.update(seen)

    min_docs = n_docs // 200
    kw_text = {kw for kw, n in counts.items() if n > min_docs}
    english = set(nltk_stopwords.words("english"))
    kw_ws = {kw.split()[0] for kw in kw_text} - english
    return kw_text, kw_ws


class FancyTokenizer:
    """Callable, POS-aware tokenizer carrying the keyword vocabulary.

    Instantiated once with the corpus keyword vocab and handed to
    sklearn's TfidfVectorizer(tokenizer=...); __call__ is a generator, which
    the vectorizer consumes per document.
    """

    def __init__(self, kw_text: set, kw_ws: set) -> None:
        self.kw_text = kw_text
        self.kw_ws = kw_ws
        self.stopwords = set(nltk_stopwords.words("english")) | EXTRA_STOPWORDS
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text: str) -> Iterable[str]:
        text = text or ""

        # Pass 1: merge known keyword phrases, removing them from the text.
        # Iteration is sorted (not set-order) so overlapping-phrase
        # substitution is reproducible across runs — the vectorizer feeds a
        # seeded NMF, and we do not want hash order leaking into the matrix.
        present_first_words = sorted(set(text.lower().split()) & self.kw_ws)
        for w in present_first_words:
            w = w.replace("(", "").replace(")", "")
            pattern = f"({re.escape(w)}\\W*\\w*)"
            found = sorted(
                {
                    m.lower().replace("-", " ")
                    for m in re.findall(pattern, text, re.IGNORECASE)
                }
                & self.kw_text
            )
            for match in found:
                text = re.sub(re.escape(match), " ", text, flags=re.IGNORECASE)
                yield match.replace(" ", "-")

        # Pass 2: POS-aware lemmatisation of the remaining text.
        for sent in sent_tokenize(text):
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                token = token.lower().strip()
                if token in self.stopwords:
                    continue
                if all(c in _PUNCT for c in token):
                    continue
                if len(token) < 3:
                    continue
                if all(c.isdigit() for c in token):
                    continue
                yield self.lemmatizer.lemmatize(token, _TAG_MAP.get(tag[0], wn.NOUN))


def build_vectorizer(kw_text: set, kw_ws: set, min_df: int = 150) -> TfidfVectorizer:
    """TfidfVectorizer wired to the fancy tokenizer, matching 2020 settings.

    max_df=0.9 and single-grams as in do_nmf; min_df scales with corpus size
    (150 for ~1.5-2m docs vs the original 50 at 400k).
    """
    return TfidfVectorizer(
        tokenizer=FancyTokenizer(kw_text, kw_ws),
        preprocessor=None,
        max_df=0.9,
        min_df=min_df,
        ngram_range=(1, 1),
        stop_words=sorted(set(nltk_stopwords.words("english")) | EXTRA_STOPWORDS),
    )
