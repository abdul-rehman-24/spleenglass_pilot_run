"""E4 design (deterministic). Generates the 12 subsets fixed in PREREGISTRATION_addendum3.md.
Inputs: sav_table.csv (32-volume pool), the 19 random-cloud subsets, E2, CT-FM and SA/V subsets (all from the logs)."""
import json, glob, random, itertools
import numpy as np, pandas as pd

SAV = "/mnt/user-data/uploads/sav_table.csv"
sv = pd.read_csv(SAV); sv["vol"] = sv.volume_mm3 / 1000
x, y = np.log(sv.vol), np.log(sv.sav)
slope, intercept = np.polyfit(x, y, 1)
sv["res"] = y - (intercept + slope * x)               # size-controlled boundary measure (residual log SA/V), fitted on all 32 pool volumes
sv = sv.set_index("volume_id")
pool = sorted(sv.index, key=lambda v: int(v.split("_")[1]))
assert len(pool) == 32

# subsets already used (as sets): random cloud (19), CT-FM, SA/V, E2
used = {}
for f in glob.glob("/tmp/e1/raw_logs/*.json") + glob.glob("/tmp/ext/raw_logs/*.json") + glob.glob("/tmp/v2/raw_logs/*N05*attentionunet*.json"):
    d = json.load(open(f)); used[frozenset(d["selected_ids"])] = d["run_id"]
cloud = [frozenset(d["selected_ids"]) for f in glob.glob("/tmp/e1/raw_logs/e1_*.json") + glob.glob("/tmp/ext/raw_logs/*.json")
         for d in [json.load(open(f))]] + [frozenset(json.load(open(f))["selected_ids"]) for f in glob.glob("/tmp/v2/raw_logs/v2_N05_random_attentionunet_seed*.json")]
cloud = list(set(cloud)); assert len(cloud) == 19, len(cloud)
e2 = frozenset(json.load(open(glob.glob("/tmp/e1/raw_logs/e2_*tr42.json")[0]))["selected_ids"])
mv = lambda s: float(sv.loc[list(s)].vol.mean()); mr = lambda s: float(sv.loc[list(s)].res.mean())
cloud_res = [mr(s) for s in cloud]; lo_res, hi_res = min(cloud_res), max(cloud_res)
E2_VOL = mv(e2)
VOL_LO, VOL_HI = 0.9 * E2_VOL, 1.1 * E2_VOL

def gen(seed):                                          # same shuffle rule as gen_random
    sh = sorted(pool, key=lambda v: int(v.split("_")[1])); random.Random(seed).shuffle(sh); return sh[:5]

# E4a: volume-matched random subsets with ordinary boundary measure
e4a, seed = [], 2001
while len(e4a) < 6:
    ids = gen(seed); s = frozenset(ids)
    if VOL_LO <= mv(s) <= VOL_HI and lo_res <= mr(s) <= hi_res and len(s & e2) <= 2 and s not in used and all(s != frozenset(a[1]) for a in e4a):
        e4a.append((seed, ids))
    seed += 1
# E4b: 5 volumes drawn at random from the top-10 residual group (E2 = top 5 of it); at most 2 volumes shared with E2; mean residual above the whole random cloud
top10 = list(sv.sort_values("res", ascending=False).index[:10])
e4b, seed = [], 3001
while len(e4b) < 6:
    ids = random.Random(seed).sample(top10, 5); s = frozenset(ids)
    if len(s & e2) <= 2 and mr(s) > hi_res and s not in used and all(s != frozenset(a[1]) for a in e4b):
        e4b.append((seed, ids))
    seed += 1

if __name__ == "__main__":
    print(f"E2 mean volume {E2_VOL:.1f} mL -> E4a window [{VOL_LO:.1f}, {VOL_HI:.1f}] mL; cloud mean-residual range [{lo_res:.3f}, {hi_res:.3f}]; E2 mean residual {mr(e2):.3f}")
    print("top-10 residual group:", [(v.replace('spleen_',''), round(sv.loc[v].res,3), round(sv.loc[v].vol)) for v in top10])
    for name, L in (("E4a", e4a), ("E4b", e4b)):
        print(f"\n{name}:")
        for sd, ids in L:
            s = frozenset(ids); print(f"  seed {sd}: {', '.join(i.replace('spleen_','') for i in ids):22s} mean vol {mv(s):6.1f} mL | mean residual {mr(s):+.3f} | overlap with E2: {len(s & e2)}")
    allv = [mv(frozenset(i)) for _, i in e4a] ; allb=[mv(frozenset(i)) for _, i in e4b]
    print("\nE4a mean volumes %.0f-%.0f ; E4b mean volumes %.0f-%.0f ; cloud volumes 144-343"%(min(allv),max(allv),min(allb),max(allb)))
    print("residual: E4a within cloud range:", all(lo_res<=mr(frozenset(i))<=hi_res for _,i in e4a), "| E4b min mean residual %.3f"%min(mr(frozenset(i)) for _,i in e4b))
