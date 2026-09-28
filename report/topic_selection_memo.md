# Choosing K for the 2026 topic model

*Evidence assembled 2026-09-25/28: grid extension, coherence, ladder, 2019
diff, and the K200 document audit (1.44M abstracts).*

**Decision: K = 200 (α = 0), fit on the deterministic 200k-document sample and
applied to all 1.44M abstracts.** Rationale below; caveats stated honestly.

## Why the old (2019-inherited) window had to be reopened

The published 2019 model runs **K = 140 on ~400k documents**, chosen near the
top of its own 80–150 grid (confirmed from three independent traces in the
recovered project: `topics.csv` with 140 rows; RunStats pk 1861 appearing in
five analysis tables; the `run_1861_topics_140` ladder sheet). The 2026 sweep
initially re-used that grid. On 3.6× more literature, evaluating K only inside
an inherited window risks re-choosing the 2019 ceiling by default. We extended
the grid to K = 220 (80–150 step 10, then 160–220 step 20, α = 0) and re-ran
all selection diagnostics over the full range.

## Evidence axes

**Coherence (NPMI over the full 1.44M-abstract corpus).** Mean topic coherence
is flat across the entire grid (0.205–0.215) and **zero topics are incoherent
at any K** — a structural soundness statement in itself. The discriminating
signal is the weakest-decile (p10) tail: 0.137 (K100) → 0.130 (K110) →
0.108–0.118 (K120–150) → 0.101–0.112 (K160–220). The tail drops crossing
~120–150 and then **plateaus**: going high costs worst-case word coherence
once, and nothing thereafter. K200's mean (0.215) is the grid maximum.

**Genealogical stability (ladder).** Adjacent-run top-word continuity dips at
K110→120 (7.2/10) and K140→150 (7.1/10) — the K120–150 band is genuinely
turbulent — and recovers to 7.4–7.5 for all rungs above 160. Splits per added
topic are *lowest* in the K160–220 range (~14–15 per +10 K, vs 19–23 in the
130–150 band): at high K the model refines, at mid-K it thrashes.

**Continuity with the 2019 published model.** For each of the 140 published
topics, best top-word overlap with the new grid: 2019 topics "tracked" rise
monotonically from 88/140 (K100) to **132/140 (K200)**; mean best overlap dips
to its grid minimum exactly at K140 (5.12) and peaks at K200 (5.86). Caveat,
stated plainly: a larger candidate pool mechanically raises max-overlap, so
read this as "K200 matches 2019 at least as well as K140 does", not as a
strict 5.86 > 5.12. (The K140 dip proves the metric is not purely a
selection artifact — 140 candidates score worse than 110 candidates.)

**Topic census.** No dust topics anywhere: smallest topics at K160–220 are
~21–24 soft-assigned docs of 200k, the same scale as K80.

## K = 140 is not the safe longitudinal choice

Under all three axes the K120–150 band is the worst region of the grid, and a
word-level diff against the 2019 topics shows K140 *dissolves* real 2019
structure: the WG1 atmospheric-composition cluster (N2O, CH4, ozone,
absorption, concentrations — five separate 2019 topics) merges into mixed
chemistry topics, and species-distribution modelling vanishes into a generic
biodiversity topic. K200 *re-separates* both (own topics for stratospheric
ozone and for habitat-suitability modelling) while keeping everything K140
gained.

## What K200 adds over K140

K140 already captures the post-Paris sociotechnical layer — batteries/EV,
machine-learning forecasting, supply-chain sustainability, green finance,
smart cities, building physics, anaerobic digestion, wastewater, nuclear
safety-society, disaster resilience, CCS, carbon pricing, blue carbon,
international law (~30 topics with no 2019 counterpart). K200 keeps that and
adds a second tier of ~40 further subfields: wildfire, permafrost, gas
hydrates, cement/clinker, plastics, tourism, education, transport mobility,
microgrid & demand response, grazing systems, alpine/plateau ecology, island
UHI, algal blooms, public perception, digital economy — plus finer splits of
K140 topics.

**The new topics are load-bearing.** Document-level scores (all 1.44M abstracts)
put the 68 no-2019-counterpart topics at 27.7% of total topic mass in
2015–19 and **31.5% in 2021–25** — growing, while matched topics lose share.
Of the 68, 9 grew and 59 held flat since 2015–19; **none is shrinking**. Their
median size (5,549 docs) is 80% of the matched topics' — they are normal-sized
literatures, not fragments. Close readings confirm identity: the SDM split
(T149) carries MaxEnt habitat-suitability papers ("Prediction of potential
suitable habitats for *Limosa limosa* in China…"), T196 carries stratospheric
ozone-depletion-and-recovery work, T34 wildfire risk/health, T129 cement
clinker chemistry, T134 grid-integration engineering.

**No dust at any scale.** The smallest topic in the full corpus (mangrove/blue
carbon, 2,252 docs) still returns exactly its own literature; tourism, biogas,
wetland restoration and nuclear security fill 2.7k–3k-doc topics. The "size 4"
values in the fit-sample census are sample-scale artifacts (~40× corpus), not
dust.

## What no K recovers

2019's generic discourse topics ("increase/decrease", "methods", "time",
"global", "seasons", "causality", ~10–15% of the 2019 corpus) have no topic
of their own at any 2026 K. The writing didn't disappear — in a 2× finer
domain structure it distributes across domain topics. This is a real change
in how the literature is organised, worth noting in the paper rather than
tuning K around.

## Known costs of K = 200

- p10 coherence sits on the plateau (≈0.107) rather than the K100 peak (0.137).
  Close-reading the eight weakest topics shows what that means: they are real
  but *miscellaneous* — a corrections/errata catch-all (T119, 13.5k docs, top
  loading "Correction to: …"), measurement-condition filler (T135 "high, low,
  value, density"), climate-media discourse (T53), control-theory-in-greenhouses
  (T180). Each holds 7k–14k docs (~0.5–1% of corpus). They are honest topics —
  this is what the literature's connective tissue looks like — but they should
  not headline the paper's topic table. (Incidental finding: errata/corrigenda
  are ~1% of the Scopus records and NMF gathers them into their own topics —
  worth filtering document types in any future re-fit.)
- 200 topics is more to narrate; the report table needs the cohort structure
  (2019-matched vs new) to stay readable.
- Document-level assignment stays crisp: median doc puts 18.7% of its mass on
  its top topic (37× the uniform-topic baseline) and ~2 topics above 10%;
  only **4.6% of documents are orphans** (no topic above 10%). Full
  distribution: `report/tables/audit_concentration_a200.csv`.

## Reproduction

Runs: `data/topics/runs/K{80..220}_a0.0` (DVC). Selection sheets:
`report/tables/` (ladder, coherence, baselines; `python -m
climate_literature.topics.compare ladder --alpha 0`, `python -m
climate_literature.topics.coherence --alpha 0`). Released artifact:
`data/topics/models/K200_a0.0` (finalize: joblib + plain-format fitted state +
golden set; `train verify` proves reproduction). Document audit over
`data/topics/doc_topics/K200_a0.0` (DVC): `scripts/topics_audit_docs.py`
(writes `report/tables/audit_{series,cohort_growth,sizes,concentration}_a200.csv`)
and `scripts/topics_audit_readings.py` (close-reading digest, stdout).
