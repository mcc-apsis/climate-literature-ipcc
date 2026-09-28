"""K200_a0.0 document audit: concentration, orphans, annual series, cohorts.

Inputs: data/topics/doc_topics/K200_a0.0 (apply output), report/tables/
(coherence_detail + baseline for cohorts), data/predictions (per-item meta).
Outputs: report/tables/audit_*.csv + printed digest for the memo.
"""

import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TAG = "K200_a0.0"
K = 200
DT_DIR = f"data/topics/doc_topics/{TAG}"
ORPHAN = 0.1  # max-topic score below this = doc no topic claims

# --- load doc-topic scores (topic_0..K-1 as float32) -----------------------
shards = sorted(glob.glob(f"{DT_DIR}/shard=*/*.parquet"))
assert shards, f"no scores under {DT_DIR}"
cols = ["item_id"] + [f"topic_{t}" for t in range(K)]
H = pd.concat(
    pd.read_parquet(f, columns=cols).set_index("item_id") for f in shards
).astype(np.float32)
n = len(H)
exp = json.load(open(f"{DT_DIR}/meta.json"))["n_docs_apply"]
assert n == exp, f"only {n:,} of {exp:,} scored docs loaded — checkout still running?"
print(f"docs: {n:,}  topics: {K}")

V = H.to_numpy(copy=True)
rs = V.sum(axis=1, keepdims=True)
print(f"zero-mass docs: {(rs[:, 0] == 0).sum()}")
V /= np.maximum(rs, 1e-9)  # each doc's loadings sum to 1 (NMF already
# does this up to fit noise — renormalise for clean entropy)

# --- per-doc concentration & orphans ---------------------------------------
mx = V.max(axis=1)
srt = -np.sort(-V, axis=1)
top1, top2 = srt[:, 0], srt[:, 1]
P = np.clip(V, 1e-9, None)
ent = -(P * np.log(P)).sum(1) / np.log(K)  # 0=one topic, 1=uniform
n_10 = (V >= 0.1).sum(1)
doc = pd.DataFrame(
    {"max_share": mx, "top2": top1 + top2, "entropy": ent, "n_ge_10pct": n_10},
    index=H.index,
)
doc["orphan"] = mx < ORPHAN
print("\n== concentration (per-doc share on its top topic) ==")
print(doc[["max_share", "top2", "entropy", "n_ge_10pct"]].describe().round(3))
print(f"orphans (max<{ORPHAN}): {doc['orphan'].mean():.2%}")

# --- topic sizes ------------------------------------------------------------
sizes = V.sum(0)
size = pd.DataFrame({"topic": np.arange(K), "docs_equiv": sizes.round(1)})

# --- cohorts: 2019-matched vs new (baseline sheet, overlap<3 = new) ---------
base = pd.read_csv(f"report/tables/baseline_{TAG}_vs_2019.csv")
base["topic_b"] = base["topic_b"].astype(int)
new_t = set(base.loc[base.overlap < 3, "topic_b"])
matched = {
    int(r.topic_b): int(r.topic_a)
    for _, r in base[base.overlap >= 3].sort_values("overlap").iterrows()
}  # later rows (higher overlap) overwrite -> best 2019 parent
coh = np.array(["new" if t in new_t else "2019-matched" for t in range(K)])
print(
    f"\ncohorts: {int((coh == 'new').sum())} new / {int((coh != 'new').sum())} matched"
)
print(
    "median size: new",
    int(size.docs_equiv[coh == "new"].median()),
    " matched",
    int(size.docs_equiv[coh != "new"].median()),
)

# --- corpus meta join (year, relevant) --------------------------------------
metas = []
for f in glob.glob("data/predictions/**/*.parquet", recursive=True):
    m = pq.read_table(
        f, columns=["item_id", "publication_year", "relevant"]
    ).to_pandas()
    metas.append(m)
meta = pd.concat(metas).drop_duplicates("item_id").set_index("item_id")
common = doc.index.intersection(meta.index)
print(f"joined year meta: {len(common):,} of {n:,} docs")
Vc = V[doc.index.get_indexer(common)]
yv = meta["publication_year"].loc[common].to_numpy()
ok = (yv >= 1990) & (yv <= 2025)
Vc, yv = Vc[ok], yv[ok]
print(f"in 1990-2025 window: {ok.sum():,}")

# --- annual series per topic -------------------------------------------------
# share of year y's literature carried by topic t (mean loading across docs of y)
years = np.arange(1990, 2026)
series = np.zeros((K, len(years)))
cnt = np.zeros(len(years))
for i, y in enumerate(years):
    sel = yv == y
    cnt[i] = sel.sum()
    series[:, i] = Vc[sel].mean(axis=0) * cnt[i]  # topic volume per year
ser = pd.DataFrame(
    series,
    index=[f"T{t}" for t in range(K)],
    columns=[f"y{y}" for y in years],
).T
ser["docs"] = cnt.astype(int)
ser.to_csv("report/tables/audit_series_a200.csv")

# cohort shares over time + recent growth for load-bearing test
sh = pd.DataFrame(series.T, index=years, columns=range(K)) / np.maximum(cnt, 1)[:, None]
new_cols = [t for t in range(K) if coh[t] == "new"]
match_cols = [t for t in range(K) if coh[t] != "new"]
early = sh.loc[1990:2014]  # not used for ratio, kept for context
lb = pd.DataFrame(
    {
        "topic": new_cols + match_cols,
        "cohort": np.array(coh)[new_cols + match_cols],
        "share_2015_19": sh.loc[2015:2019][new_cols + match_cols].sum().to_numpy(),
        "share_2021_25": sh.loc[2021:2025][new_cols + match_cols].sum().to_numpy(),
    }
)
print("\n== cohort share of literature (sum of topic shares) ==")
print(lb.groupby("cohort")[["share_2015_19", "share_2021_25"]].sum().round(3))
grow = lb[lb.cohort == "new"].copy()
grow["ratio"] = (grow.share_2021_25 + 1e-3) / (grow.share_2015_19 + 1e-3)
print(
    f"new topics: {(grow.ratio > 1.5).sum()} growing / "
    f"{(grow.ratio < 0.7).sum()} shrinking / "
    f"{((grow.ratio >= 0.7) & (grow.ratio <= 1.5)).sum()} flat "
    f"(of {len(grow)})"
)
lb.to_csv("report/tables/audit_cohort_growth_a200.csv", index=False)

# weakest p10 coherence topics for the close-reading list
det = pd.read_csv("report/tables/coherence_detail_a0.0.csv")
d200 = det[det.tag == TAG].nsmallest(8, "npmi")
print("\n== 8 weakest-coherence K200 topics ==")
for _, r in d200.iterrows():
    t = int(r.topic)
    print(f"  T{t:3} npmi {r.npmi:.3f} size {r.size:.0f} new={coh[t] == 'new'}")

size.to_csv("report/tables/audit_sizes_a200.csv", index=False)
doc[["max_share", "top2", "entropy", "n_ge_10pct", "orphan"]].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
).round(4).T.to_csv("report/tables/audit_concentration_a200.csv")
print("\nwrote audit_series / audit_cohort_growth / audit_sizes / audit_concentration")
print(json.dumps({"n": int(n), "orphan_rate": round(float(doc.orphan.mean()), 4)}))
