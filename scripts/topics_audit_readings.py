"""Stage 2: close readings — top docs + trajectory for selected K200 topics.

Reads the doc-topic scores once; prints, per chosen topic: top words, size,
share trajectory (first-decade vs recent share), and the titles of the 3
most-characteristic docs. Topics: the predicted-important splits, the 8
weakest coherence, the 5 largest, the 5 smallest, and 3 highest-entropy-topic
docs (what orphan-ish docs look like).
"""

import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from climate_literature.topics.compare import RUNS_DIR, _load_run

TAG = "K200_a0.0"
K = 200
DT = f"data/topics/doc_topics/{TAG}"

H = pd.concat(
    pd.read_parquet(
        f, columns=["item_id"] + [f"topic_{t}" for t in range(K)]
    ).set_index("item_id")
    for f in sorted(glob.glob(f"{DT}/shard=*/*.parquet"))
).astype(np.float32)
V = H.to_numpy(copy=True)
V /= np.maximum(V.sum(axis=1, keepdims=True), 1e-9)

metas = [
    pq.read_table(
        f, columns=["item_id", "title", "publication_year", "relevant"]
    ).to_pandas()
    for f in glob.glob("data/predictions/**/*.parquet", recursive=True)
]
meta = pd.concat(metas).drop_duplicates("item_id").set_index("item_id")
common = H.index.intersection(meta.index)
pos = H.index.get_indexer(common)
Vc, M = V[pos], meta.loc[common]
yv = M["publication_year"].to_numpy()

words = _load_run(RUNS_DIR / TAG).set_index("topic")["words"]

det = pd.read_csv("report/tables/coherence_detail_a0.0.csv")
weak = det[det.tag == TAG].nsmallest(8, "npmi")["topic"].astype(int).tolist()
sz = Vc.sum(0)
big = np.argsort(-sz)[:5].tolist()
small = np.argsort(sz)[:5].tolist()
chosen = {
    "predicted splits": [149, 196, 34, 129, 134],
    "weakest coherence": weak,
    "5 largest": big,
    "5 smallest": small,
}
seen = set()
for group, ts in chosen.items():
    print(f"\n########## {group}")
    for t in ts:
        if t in seen:
            continue
        seen.add(t)
        col = Vc[:, t]
        y_mask = (yv >= 1998) & (yv <= 2025)
        s98 = col[y_mask & (yv <= 2010)].sum() / max((y_mask & (yv <= 2010)).sum(), 1)
        s16 = col[y_mask & (yv >= 2016)].sum() / max((y_mask & (yv >= 2016)).sum(), 1)
        top3 = np.argsort(-col)[:3]
        print(f"\nT{t}  size {sz[t]:7.0f}  share<=2010 {s98:.4f} -> >=2016 {s16:.4f}")
        print(f"  words: {words.loc[t]}")
        for i in top3:
            ttl = str(M["title"].iloc[i])[:110]
            print(f"  {int(yv[i])} {col[i]:.2f} | {ttl}")
