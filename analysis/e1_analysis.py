"""Pre-registered E1/E2 analysis (PREREGISTRATION.md @ e092150 + addendum 1 @ 8081e7b).
Run:  python e1_analysis.py  (expects e1 logs, v2 logs and sav_table.csv; paths below)."""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy import optimize, stats

E1_DIR, V2_DIR, SAV = "/tmp/e1/raw_logs", "/tmp/v2/raw_logs", "/mnt/user-data/uploads/sav_table.csv"
RNG = np.random.default_rng(20261028)
NBOOT = 5000

# ------------------------------------------------------------------ load
def load(path):
    d = json.load(open(path))
    cur = d["val_dsc_curve"]
    return dict(run_id=d["run_id"], strategy=d["strategy"], sel=d.get("selection_seed"), tr=d["training_seed"],
                best=100 * d["best_val_dsc"], last5=100 * float(np.mean(cur[-5:])), epochs=d["epochs_run"],
                ids=tuple(d["selected_ids"]), pervol=d["best_per_volume_val_dsc"])
e1 = [load(f) for f in sorted(glob.glob(f"{E1_DIR}/*.json"))]
v2 = {os.path.basename(f)[:-5]: load(f) for f in glob.glob(f"{V2_DIR}/v2_N05_*_attentionunet_seed*.json")}
assert len(e1) == 19 and len(v2) == 9

rows = []
for r in e1:
    if r["run_id"].startswith("e1_"):
        rows.append(dict(r, group=r["sel"], diag=(r["sel"] == r["tr"]), kind="random"))
for s in (42, 123, 7):                       # diagonal runs of the three existing random subsets
    r = v2[f"v2_N05_random_attentionunet_seed{s}"]; assert r["sel"] == s and r["tr"] == s
    rows.append(dict(r, group=s, diag=True, kind="random"))
R = pd.DataFrame(rows)
assert len(R) == 19 and R.group.nunique() == 13
# the reseeded runs must use the same volumes as the diagonal run of their subset
for g, sub in R.groupby("group"):
    assert len({x for x in sub.ids}) == 1, g
E2 = pd.DataFrame([r for r in e1 if r["run_id"].startswith("e2_")])
CT = pd.DataFrame([v2[f"v2_N05_ctfmdiverse_attentionunet_seed{s}"] for s in (42, 123, 7)])
SV = pd.DataFrame([v2[f"v2_N05_savhighest_attentionunet_seed{s}"] for s in (42, 123, 7)])

# ------------------------------------------------------------------ REML (one-way random effects)
def reml(groups):
    """groups: list of 1-D arrays. Returns (sigma_S2, sigma_e2). Profile REML over tau = sigma_S2/sigma_e2 >= 0."""
    n = np.array([len(g) for g in groups], float); N = n.sum()
    ybar = np.array([g.mean() for g in groups]); W = np.array([((g - g.mean()) ** 2).sum() for g in groups])
    def parts(tau):
        w = n / (1 + n * tau); mu = (w * ybar).sum() / w.sum()
        Q = W.sum() + (w * (ybar - mu) ** 2).sum()
        return Q / (N - 1), w
    def nll(tau):
        s2, w = parts(tau)
        return (N - 1) * np.log(s2) + np.log(1 + n * tau).sum() + np.log(w.sum())
    grid = np.concatenate([[0.0], np.logspace(-4, 3, 120)])
    vals = [nll(t) for t in grid]; k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    tau = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded").x if hi > lo else 0.0
    if nll(0.0) <= nll(tau) + 1e-12: tau = 0.0
    s2, _ = parts(tau)
    return tau * s2, s2

def anova(groups):
    n = np.array([len(g) for g in groups], float); N = n.sum(); K = len(groups)
    allm = np.concatenate(groups).mean()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups); ssb = sum(len(g) * (g.mean() - allm) ** 2 for g in groups)
    msw, msb = ssw / (N - K), ssb / (K - 1); n0 = (N - (n ** 2).sum() / N) / (K - 1)
    return max(0.0, (msb - msw) / n0), msw

def share(sS, se): return sS / (sS + se) if (sS + se) > 0 else np.nan

def groups_of(col): return [g[col].values.astype(float) for _, g in R.groupby("group")]

out = {}
for col in ("best", "last5"):
    G = groups_of(col)
    sS, se = reml(G); aS, ae = anova(G)
    boots = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(G), len(G)); b = [G[i] for i in idx]
        if all(len(x) == 1 for x in b): continue          # cannot estimate sigma_e from singletons only
        s1, e1_ = reml(b); boots.append(share(s1, e1_))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out[col] = dict(reml_sigma_S=np.sqrt(sS), reml_sigma_e=np.sqrt(se), reml_share=share(sS, se), ci=(lo, hi),
                    anova_sigma_S=np.sqrt(aS), anova_sigma_e=np.sqrt(ae), anova_share=share(aS, ae),
                    boot_share_zero=float(np.mean(np.array(boots) == 0)), n_boot=len(boots))

# ------------------------------------------------------------------ training-noise estimates (reported separately)
def within_sd(df_groups):
    ss = sum(((g - g.mean()) ** 2).sum() for g in df_groups); dof = sum(len(g) - 1 for g in df_groups)
    return np.sqrt(ss / dof), dof
noise = {}
for col in ("best", "last5"):
    resampled = [g[col].values for _, g in R.groupby("group") if len(g) == 3]
    sd_r, dr = within_sd(resampled)
    sd_c, dc = within_sd([CT[col].values]); sd_s, ds = within_sd([SV[col].values])
    pooled, dp = within_sd(resampled + [CT[col].values, SV[col].values])
    noise[col] = dict(random_reseeded=(sd_r, dr), ctfm=(sd_c, dc), sav=(sd_s, ds), pooled=(pooled, dp))

# ------------------------------------------------------------------ Rules 2 and 3
def rank_desc(value, cloud):           # rank among cloud+value, 1 = highest
    return 1 + int(np.sum(np.array(cloud) > value))
def pstar_upper(value, cloud): return (1 + int(np.sum(np.array(cloud) >= value))) / (len(cloud) + 1)
def pstar_lower(value, cloud): return (1 + int(np.sum(np.array(cloud) <= value))) / (len(cloud) + 1)

rule = {}
for col in ("best", "last5"):
    cloud_means = R.groupby("group")[col].mean().values                 # 13 subset means (primary)
    cloud_diag = R[R.diag].sort_values("group")[col].values             # 13 single runs (addendum A2)
    assert len(cloud_means) == 13 and len(cloud_diag) == 13
    res = {}
    for name, df in (("ctfm", CT), ("sav", SV), ("e2", E2)):
        m = df[col].mean(); rk = rank_desc(m, cloud_means)
        per_seed = [rank_desc(v, cloud_diag) for v in df[col].values]
        res[name] = dict(mean=m, rank14=rk, pstar_up=pstar_upper(m, cloud_means), pstar_low=pstar_lower(m, cloud_means),
                         per_seed_values=list(df[col].values), per_seed_ranks=per_seed, median_rank=float(np.median(per_seed)))
    res["cloud"] = dict(means=sorted(cloud_means), mean_of_means=float(np.mean(cloud_means)), median=float(np.median(cloud_means)),
                        min=float(np.min(cloud_means)), max=float(np.max(cloud_means)), diag_sorted=sorted(cloud_diag))
    rule[col] = res

# ------------------------------------------------------------------ Rule 4 (exploratory): subset mean volume vs DSC
sv = pd.read_csv(SAV); vol = dict(zip(sv.volume_id, sv.volume_mm3 / 1000))
R["subset_mean_vol"] = R.ids.apply(lambda ids: np.mean([vol[i] for i in ids]))
sub = R.groupby("group").agg(vol=("subset_mean_vol", "first"), best=("best", "mean"), last5=("last5", "mean")).reset_index()
rule4 = {c: stats.spearmanr(sub.vol, sub[c]) for c in ("best", "last5")}
diag = R[R.diag].sort_values("group"); rule4_diag = {c: stats.spearmanr(diag.subset_mean_vol, diag[c]) for c in ("best", "last5")}
vol_det = dict(ctfm=np.mean([vol[i] for i in CT.ids[0]]), sav=np.mean([vol[i] for i in SV.ids[0]]), e2=np.mean([vol[i] for i in E2.ids[0]]))

# ------------------------------------------------------------------ MDE arithmetic (descriptive)
def mde(sS, se, K, Rstar, Rcloud=1.0):
    return 2.8 * np.sqrt(se ** 2 / Rstar + sS ** 2 / K + se ** 2 / (K * Rcloud))

pd.set_option("display.width", 200)
import pickle
pickle.dump(dict(out=out, noise=noise, rule=rule, rule4=rule4, rule4_diag=rule4_diag, vol_det=vol_det, R=R, sub=sub, E2=E2, CT=CT, SV=SV), open("/home/claude/an/results.pkl", "wb"))

# ------------------------------------------------------------------ report
def f(x, d=2): return f"{x:.{d}f}"
print("RUNS: 19 random-subset runs (13 subsets); deterministic: CT-FM 3, SA/V 3 (v2); E2 3\n")
print("=== Rule 1: variance split (DSC points, Attention U-Net, N=5) ===")
for col in ("best", "last5"):
    o = out[col]
    print(f"[{col}] REML: sigma_S={f(o['reml_sigma_S'])}  sigma_xi={f(o['reml_sigma_e'])}  share={f(100*o['reml_share'],1)}%  95% CI [{f(100*o['ci'][0],1)}, {f(100*o['ci'][1],1)}]%  (bootstrap share==0 in {f(100*o['boot_share_zero'],1)}% of {o['n_boot']})")
    print(f"       ANOVA: sigma_S={f(o['anova_sigma_S'])}  sigma_xi={f(o['anova_sigma_e'])}  share={f(100*o['anova_share'],1)}%")
print("\n=== Training noise estimates (SD, DSC points; df) ===")
for col in ("best", "last5"):
    n = noise[col]; print(f"[{col}] reseeded random subsets {f(n['random_reseeded'][0])} (df {n['random_reseeded'][1]}) | CT-FM {f(n['ctfm'][0])} (2) | SA/V {f(n['sav'][0])} (2) | pooled {f(n['pooled'][0])} (df {n['pooled'][1]})")
print("\n=== Rules 2 and 3 ===")
for col in ("best", "last5"):
    r = rule[col]; c = r["cloud"]
    print(f"[{col}] cloud of 13 subset means: min {f(c['min'])} median {f(c['median'])} mean {f(c['mean_of_means'])} max {f(c['max'])}")
    for nm in ("ctfm", "sav", "e2"):
        x = r[nm]
        print(f"   {nm:5s} mean {f(x['mean'])} | rank among 14 (1=highest) {x['rank14']} | p*_upper {f(x['pstar_up'],3)} p*_lower {f(x['pstar_low'],3)} | single-run values {[f(v) for v in x['per_seed_values']]} ranks {x['per_seed_ranks']} median {x['median_rank']}")
print("\n=== Rule 4 (exploratory): Spearman(subset mean volume, DSC) over 13 subsets ===")
for col in ("best", "last5"):
    print(f"[{col}] subset means: rho={f(rule4[col][0])} p={f(rule4[col][1],3)} | diagonal runs: rho={f(rule4_diag[col][0])} p={f(rule4_diag[col][1],3)}")
print("mean volume (mL): CT-FM", f(vol_det['ctfm'],0), "| SA/V", f(vol_det['sav'],0), "| E2", f(vol_det['e2'],0), "| random subsets range", f(sub.vol.min(),0), "-", f(sub.vol.max(),0))


# ------------------------------------------------------------------ extras (run-level table, ANOVA interval, MDE) - descriptive
print("\n=== Run level (DSC points) ===")
R["gap"] = R.best - R.last5
print(R.sort_values(["group", "tr"])[["group", "tr", "best", "last5", "gap", "epochs", "subset_mean_vol"]].round(2).to_string(index=False))
print("mean best-minus-last5 gap: random runs %.2f | CT-FM %.2f | SA/V %.2f | E2 %.2f" % (R.gap.mean(), (CT.best - CT.last5).mean(), (SV.best - SV.last5).mean(), (E2.best - E2.last5).mean()))
print("\n=== ANOVA estimator with the same subset-level bootstrap ===")
for col in ("best", "last5"):
    G = groups_of(col); b = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(G), len(G)); bb = [G[i] for i in idx]
        if sum(len(x) for x in bb) - len(bb) < 1: continue
        s_, e_ = anova(bb); b.append(share(s_, e_))
    b = np.array(b)
    print("[%s] ANOVA share %.1f%% CI [%.1f, %.1f]%% ; share==0 in %.1f%%" % (col, 100 * out[col]["anova_share"], *(100 * np.nanpercentile(b, [2.5, 97.5])), 100 * np.mean(b == 0)))
print("\n=== Minimum detectable effect (80% power, 5% two-sided, 2.8 x SE; validation-volume variance NOT included) ===")
for col in ("best", "last5"):
    o = out[col]
    for lab, sS, se in [("REML", o["reml_sigma_S"], o["reml_sigma_e"]), ("ANOVA", o["anova_sigma_S"], o["anova_sigma_e"])]:
        print("[%s] %s: sigma_S %.2f sigma_xi %.2f | 3-seed strategy vs K=13 cloud: %.1f | single run vs cloud: %.1f | K=19: %.1f" % (col, lab, sS, se, mde(sS, se, 13, 3, 1.46), mde(sS, se, 13, 1, 1.0), mde(sS, se, 19, 3, 1.46)))
print("\nE2 single runs (best):", E2.best.round(2).tolist(), "| max of all 19 random runs", R.best.max().round(2))
print("E2 single runs (last5):", E2.last5.round(2).tolist(), "| max of all 19 random runs", R.last5.max().round(2))
print("E2 volumes:", E2.ids[0])
