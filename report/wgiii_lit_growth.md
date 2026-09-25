# WGIII Section 1 — literature growth

The literature on climate change has continued to grow since AR6. 
The bibliometric methods of Callaghan et al. (2020) were re-applied to an updated corpus of ~1,130,000 climate-relevant Scopus records (1985–2024).
Annual publications rose from ~64,000 in 2019 to ~150,000 in 2024, (at an annual growth rate of 19%).
385,043 papers (34% of the corpus) appeared in 2022–2024, after the AR6 literature cut-off (Figure X).
Within this corpus, a subset of (91,847 papers (8% of the corpus) were classified as climate policy relevant according to the classifier in Callaghan et al. 2024. 
Climate policy relevant papers grew from  grew from ~4,800 in 2019 to ~13,000 (2024) (growing at ≈22% per year). 
The climate policy relevant papers published between 2022 and 2024 (n = 33,594) were split across sectors as follows energy 31%; cross-sectoral 29%; transport 13%; AFOLU 10%; buildings 8%; industry 7%; waste 2%.
These figures reflect publications indexed in Scopus, which over-represents English-language journals and high-income-country institutions; they illustrate relative growth in publications rather than total global scientific knowledge.

## Not fillable from this project (left flagged)

- Non-English share of the literature: Scopus abstract records carry no
  language field, so no number can be produced; keep the qualitative
  coverage caveat above.
- Regional-evidence progress (paragraph 3): no assessment-process data here.
  The predictions do contain author-affiliation countries, so an
  affiliation-country mix over time is the natural proxy if wanted later.
- `Ford et al. 2025`, the `[link]` to contribution databases, and the
  Chapter-3 database calls: outside this repo.

## Caveats for the text

- Numbers quote through 2024 only: the corpus includes in-press
  records with future cover dates (2025–2027 exist), so raw 2025–2026 counts
  are both incomplete and inflated; the papers-by-year figure showing them
  needs a footnote or the same cutoff.
- Sector detail comes from the policy-relevant cascade only (argmax over
  seven WGIII sector scores), not from the topic model — topic-level numbers
  per K are not comparable across runs and are not quotable yet.
- Paragraphs 2–3 of the skeleton (meta-analyses, database calls, AI use,
  non-English/regional progress) are untouched by this script.

## Values and provenance

| Value | Computed |
| --- | --- |
| corpus size (unique records, 1985–2024) | 1,128,773 |
| papers 2019 | 63,870 |
| papers 2024 | 152,028 |
| growth factor 2019→2024 | 2.4× |
| compound growth | 19%/yr |
| papers 2022–2024 | 385,043 |
| share of corpus 2022–2024 | 34% |
| policy-relevant papers (total) | 91,847 |
| policy-relevant share | 8% |
| relevant CAGR 2019 → 2024 | ~4,800 → ~13,000 (≈22%/yr) |
| sector split n (2022–2024) | 33,594 |
| sector split shares | energy 31%; cross-sectoral 29%; transport 13%; AFOLU 10%; buildings 8%; industry 7%; waste 2% |

Filters: unique `item_id`; publication year 1985–2024
(cover year, from `data/predictions`); policy-relevant = `relevant` >
0.5; sector = argmax over the `"8 - …"` sector
score columns.
