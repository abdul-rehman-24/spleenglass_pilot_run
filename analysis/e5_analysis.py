"""Pre-registered E5 analysis (PREREGISTRATION_addendum4.md, section E3, A-D).
Models: Attention U-Net (AU) and SAMed, N = 5, the same 19 random subsets. Metrics: best-epoch DSC (primary), mean of last 5 epochs."""
import json, glob, os, itertools
import numpy as np, pandas as pd
from scipy import optimize, stats

E1_DIR, EXT_DIR, V2_DIR, E5_DIR = "/tmp/e1/raw_logs", "/tmp/ext/raw_logs", "/tmp/v2/raw_logs", "/tmp/e5col/raw_logs"
RNG = np.random.default_rng(20261031); NBOOT = 5000

def load(path):
    d = json.load(open(path)); cur = d["val_dsc_curve"]
    return dict(run_id=d["run_id"], best=100 * d["best_val_dsc"], last5=100 * float(np.mean(cur[-5:])), ids=tuple(d["selected_ids"]),
                sel=d.get("selection_seed"), tr=d["training_seed"], epochs=d["epochs_run"])

# ---------------- runs of both models: one row per run, group = subset ----------------
def build(model):
    rows = []
    if model == "AU":
        for f in sorted(glob.glob(f"{E1_DIR}/e1_*.json")): r = load(f); rows.append(dict(r, group=r["sel"]))
        for f in sorted(glob.glob(f"{EXT_DIR}/*.json")): r = load(f); rows.append(dict(r, group=r["sel"]))
        for s in (42, 123, 7): r = load(f"{V2_DIR}/v2_N05_random_attentionunet_seed{s}.json"); rows.append(dict(r, group=s))
    else:
        for f in sorted(glob.glob(f"{E5_DIR}/e5_N05_random_samed_*.json")): r = load(f); rows.append(dict(r, group=r["sel"]))
        for s in (42, 123, 7): r = load(f"{V2_DIR}/v2_N05_random_samed_seed{s}.json"); rows.append(dict(r, group=s))
    D = pd.DataFrame(rows); D["diag"] = D.sel == D.tr
    assert len(D) == 25 and D.group.nunique() == 19 and D.diag.sum() == 19, (model, len(D), D.group.nunique(), D.diag.sum())
    return D
RUNS = {m: build(m) for m in ("AU", "SAMed")}
SAM_DET = {n: pd.DataFrame([load(f"{V2_DIR}/v2_N05_{k}_samed_seed{s}.json") for s in (42, 123, 7)]) for n, k in (("CT-FM", "ctfmdiverse"), ("SA/V", "savhighest"))}
SAM_E2 = pd.DataFrame([load(f) for f in sorted(glob.glob(f"{E5_DIR}/e5_N05_residsav_samed_*.json"))]); assert len(SAM_E2) == 3
for m in RUNS:   # same subset in both models
    pass
ids_au = RUNS["AU"].groupby("group").ids.first(); ids_sa = RUNS["SAMed"].groupby("group").ids.first()
assert all(set(ids_au[g]) == set(ids_sa[g]) for g in ids_au.index), "subsets differ between the models"

# ---------------- variance components (same estimators as the Attention U-Net analysis) ----------------
def reml(groups):
    n = np.array([len(g) for g in groups], float); N = n.sum()
    ybar = np.array([g.mean() for g in groups]); W = np.array([((g - g.mean()) ** 2).sum() for g in groups])
    def parts(tau):
        w = n / (1 + n * tau); mu = (w * ybar).sum() / w.sum(); Q = W.sum() + (w * (ybar - mu) ** 2).sum(); return Q / (N - 1), w
    def nll(tau):
        s2, w = parts(tau); return (N - 1) * np.log(s2) + np.log(1 + n * tau).sum() + np.log(w.sum())
    grid = np.concatenate([[0.0], np.logspace(-4, 3, 120)]); vals = [nll(t) for t in grid]; k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    tau = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded").x if hi > lo else 0.0
    if nll(0.0) <= nll(tau) + 1e-12: tau = 0.0
    s2, _ = parts(tau); return tau * s2, s2
def anova(groups):
    n = np.array([len(g) for g in groups], float); N = n.sum(); K = len(groups); allm = np.concatenate(groups).mean()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups); ssb = sum(len(g) * (g.mean() - allm) ** 2 for g in groups)
    msw, msb = ssw / (N - K), ssb / (K - 1); n0 = (N - (n ** 2).sum() / N) / (K - 1); return max(0.0, (msb - msw) / n0), msw
share = lambda sS, se: sS / (sS + se) if (sS + se) > 0 else np.nan
def groups_of(D, col): return [g[col].values.astype(float) for _, g in D.groupby("group")]

# A. variance split ------------------------------------------------------------------------------
A = {}
for m in RUNS:
    for col in ("best", "last5"):
        G = groups_of(RUNS[m], col); sS, se = reml(G); aS, ae = anova(G); boots = []
        for _ in range(NBOOT):
            idx = RNG.integers(0, len(G), len(G)); b = [G[i] for i in idx]
            if all(len(x) == 1 for x in b): continue
            s1, e1 = reml(b); boots.append(share(s1, e1))
        A[(m, col)] = dict(sS=np.sqrt(sS), se=np.sqrt(se), share=share(sS, se), ci=np.nanpercentile(boots, [2.5, 97.5]), zero=float(np.mean(np.array(boots) == 0)),
                           a_sS=np.sqrt(aS), a_se=np.sqrt(ae), a_share=share(aS, ae), nb=len(boots), R=share(sS, se))
def branch(a):
    lo, hi = a["ci"]; pt = a["share"]
    if lo > 0: return "branch 1 (subset matters as much or more)" if pt >= 0.5 else "branch 2 (measurable but smaller share)"
    return "branch 3 (mostly training noise)" if hi < 0.5 else "branch 4 (inconclusive, no X% claim)"

# B. single-run comparisons ------------------------------------------------------------------------
def pairs_vs_cloud(df, cloud, col):
    D = np.array([a - b for a in df[col].values for b in cloud]); return D
B = {}
for m in RUNS:
    for col in ("best", "last5"):
        cloud = RUNS[m][RUNS[m].diag].sort_values("group")[col].values
        P = np.array([a - b for a, b in itertools.permutations(cloud, 2)])
        B[(m, col, "random-vs-random")] = (np.mean(np.abs(P) > 2), np.mean(np.abs(P) > 5), np.abs(P).max())
        if m == "SAMed":
            for nm, df in list(SAM_DET.items()) + [("E2", SAM_E2)]:
                D = pairs_vs_cloud(df, cloud, col); B[(m, col, nm)] = (np.mean(D > 0), np.mean(D > 2), np.mean(D < -2), np.median(D), D.min(), D.max())

# C. E2 on SAMed ---------------------------------------------------------------------------------
rank_desc = lambda v, cloud: 1 + int(np.sum(np.array(cloud) > v))
bin20 = lambda r: "1-3 consistent with the corrected hypothesis (single-subset evidence)" if r <= 3 else ("4-10 inconclusive" if r <= 10 else "11-20 not informative")
C = {}
for col in ("best", "last5"):
    sub_means = RUNS["SAMed"].groupby("group")[col].mean().values; cloud = RUNS["SAMed"][RUNS["SAMed"].diag].sort_values("group")[col].values
    C[col] = dict(e2=SAM_E2[col].mean(), rank=rank_desc(SAM_E2[col].mean(), sub_means), ranks_single=[rank_desc(v, cloud) for v in SAM_E2[col].values],
                  cloud_median=float(np.median(sub_means)), cloud_max=float(sub_means.max()), singles=list(SAM_E2[col].values))
    C[col]["median_single"] = float(np.median(C[col]["ranks_single"]))
    for nm, df in SAM_DET.items(): C[col][nm] = (df[col].mean(), rank_desc(df[col].mean(), sub_means))

# D. transfer -----------------------------------------------------------------------------------
D_ = {}
au_d = RUNS["AU"][RUNS["AU"].diag].sort_values("group"); sa_d = RUNS["SAMed"][RUNS["SAMed"].diag].sort_values("group")
assert list(au_d.group) == list(sa_d.group)
GA = {c: groups_of(RUNS["AU"], c) for c in ("best", "last5")}; GS = {c: groups_of(RUNS["SAMed"], c) for c in ("best", "last5")}
gids = sorted(RUNS["AU"].group.unique()); assert gids == sorted(RUNS["SAMed"].group.unique())
for col in ("best", "last5"):
    x, y = au_d[col].values, sa_d[col].values
    rho = stats.spearmanr(x, y)[0]; r = stats.pearsonr(x, y)[0]
    RA, RS = A[("AU", col)]["R"], A[("SAMed", col)]["R"]; ceil = np.sqrt(RA * RS) if RA > 0 and RS > 0 else 0.0
    bs, bcorr = [], []
    for _ in range(NBOOT):
        idx = RNG.integers(0, 19, 19)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2: continue
        rb = stats.spearmanr(x[idx], y[idx])[0]; bs.append(rb)
        ga = [GA[col][i] for i in idx]; gs = [GS[col][i] for i in idx]
        if all(len(g) == 1 for g in ga) or all(len(g) == 1 for g in gs): continue
        ra = share(*reml(ga)); rs = share(*reml(gs))
        if ra > 0 and rs > 0: bcorr.append(min(1.0, rb / np.sqrt(ra * rs)))
    D_[col] = dict(rho=rho, ci=np.percentile(bs, [2.5, 97.5]), pearson=r, RA=RA, RS=RS, ceiling=ceil, corrected=(rho / ceil if ceil > 0 else np.nan),
                   corr_ci=(np.percentile(bcorr, [2.5, 97.5]) if len(bcorr) > 50 else [np.nan, np.nan]), n_corr=len(bcorr), x=x, y=y)
T = {}
last, best = D_["last5"], D_["best"]
if last["ci"][0] > 0: T["rule"] = "T1 subset quality partly shared" + (" (robust: best-epoch lower bound also > 0)" if best["ci"][0] > 0 else " (fragile: best-epoch lower bound not > 0)")
elif last["ceiling"] >= 0.6 and last["ci"][1] < 0.3: T["rule"] = "T2 weakly shared or model-specific"
else: T["rule"] = "T3 inconclusive"

# ---------------- MDE (descriptive) ----------------
mde = lambda sS, se, K, Rstar, Rcl=1.0: 2.8 * np.sqrt(se ** 2 / Rstar + sS ** 2 / K + se ** 2 / (K * Rcl))

if __name__ == "__main__":
    f = lambda v, d=2: f"{v:.{d}f}"
    print("SAMed random-subset cloud: 19 subsets, 25 runs (E5 + the 3 earlier diagonal runs); deterministic: CT-FM 3, SA/V 3 (earlier runs); E2 3 (E5)\n")
    print("=== A. Variance split (DSC points, N = 5) ===")
    for m in ("AU", "SAMed"):
        for col in ("best", "last5"):
            a = A[(m, col)]
            print(f"[{m:5s} {col:5s}] REML sigma_S {f(a['sS'])} sigma_xi {f(a['se'])} share {f(100*a['share'],1)}% CI [{f(100*a['ci'][0],1)}, {f(100*a['ci'][1],1)}]% (share==0 in {f(100*a['zero'],1)}% of {a['nb']}) | ANOVA share {f(100*a['a_share'],1)}% | -> {branch(a)}")
    print("\n=== A'. Training-noise SD of the earlier deterministic subsets (SAMed, 3 seeds each) and MDE ===")
    for col in ("best", "last5"):
        sdc, sds = SAM_DET["CT-FM"][col].std(ddof=1), SAM_DET["SA/V"][col].std(ddof=1)
        a = A[("SAMed", col)]
        print(f"[{col}] CT-FM SD {f(sdc)} | SA/V SD {f(sds)} | REML-based MDE: 3-seed strategy vs 19-subset cloud {f(mde(a['sS'], a['se'], 19, 3, 25/19),1)} | single run vs cloud {f(mde(a['sS'], a['se'], 19, 1, 1.0),1)}")
    print("\n=== B. Single-run comparisons (SAMed) ===")
    for col in ("best", "last5"):
        p = B[("SAMed", col, "random-vs-random")]; q = B[("AU", col, "random-vs-random")]
        print(f"[{col}] random-vs-random single runs: |diff|>2 in {f(100*p[0],0)}% of pairs, >5 in {f(100*p[1],0)}%, max {f(p[2],1)}   (Attention U-Net: {f(100*q[0],0)}%, {f(100*q[1],0)}%, {f(q[2],1)})")
        for nm in ("CT-FM", "SA/V", "E2"):
            v = B[("SAMed", col, nm)]; print(f"       {nm:6s} vs one random run: better in {f(100*v[0],0)}% | better by >2 in {f(100*v[1],0)}% | worse by >2 in {f(100*v[2],0)}% | median diff {v[3]:+.1f} range [{v[4]:+.1f}, {v[5]:+.1f}]")
    print("\n=== C. E2 on SAMed ===")
    for col in ("best", "last5"):
        c = C[col]; b1, b2 = bin20(c["rank"]), bin20(int(round(c["median_single"])))
        print(f"[{col}] E2 mean {f(c['e2'])} (runs {[f(v) for v in c['singles']]}) | cloud median {f(c['cloud_median'])} max {f(c['cloud_max'])} | rank of the mean among 19 subset means: {c['rank']} of 20 -> {b1}")
        print(f"       single-run ranks {c['ranks_single']} (median {c['median_single']}) -> {b2} | " + ("versions agree" if b1.split()[0] == b2.split()[0] else "versions DISAGREE -> inconclusive"))
        print(f"       CT-FM mean {f(c['CT-FM'][0])} rank {c['CT-FM'][1]} of 20 | SA/V mean {f(c['SA/V'][0])} rank {c['SA/V'][1]} of 20")
    print("\n=== D. Transfer between the two models (19 subsets, diagonal runs) ===")
    for col in ("last5", "best"):
        d = D_[col]
        print(f"[{col}{' (primary)' if col == 'last5' else ' (secondary)'}] Spearman {f(d['rho'])} 95% CI [{f(d['ci'][0])}, {f(d['ci'][1])}] | Pearson {f(d['pearson'])} | reliability AU {f(d['RA'])} SAMed {f(d['RS'])} -> ceiling {f(d['ceiling'])} | attenuation-corrected {f(d['corrected'])} (CI [{f(d['corr_ci'][0])}, {f(d['corr_ci'][1])}], n={d['n_corr']}; descriptive)")
    print("Rule outcome:", T["rule"])
    sub_ids = [int(g) for g in au_d.group]
    print("\nsubset-level DSC (diagonal runs), last5: " + "; ".join(f"{g}: AU {a:.1f} / SAMed {s:.1f}" for g, a, s in zip(sub_ids, last["x"], last["y"])))
