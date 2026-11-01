"""Pre-registered E4 analysis (PREREGISTRATION_addendum3.md @ 8531f97, section C3).
Outcome: best-epoch DSC (primary), mean of last 5 epochs (sensitivity). Cloud = 19 diagonal runs of the random subsets."""
import json, glob, os
import numpy as np, pandas as pd
from scipy import stats

E1_DIR, EXT_DIR, V2_DIR, E4_DIR = "/tmp/e1/raw_logs", "/tmp/ext/raw_logs", "/tmp/v2/raw_logs", "/tmp/e4col/raw_logs"
SAV = "/mnt/user-data/uploads/sav_table.csv"
RNG = np.random.default_rng(20261030)
NBOOT, NPERM = 5000, 10000

def load(path):
    d = json.load(open(path)); cur = d["val_dsc_curve"]
    return dict(run_id=d["run_id"], best=100 * d["best_val_dsc"], last5=100 * float(np.mean(cur[-5:])), ids=tuple(d["selected_ids"]), epochs=d["epochs_run"])

# ---------------- subsets and covariates ----------------
sv = pd.read_csv(SAV); sv["vol"] = sv.volume_mm3 / 1000
x, y = np.log(sv.vol), np.log(sv.sav); sl, ic = np.polyfit(x, y, 1)
sv["res"] = y - (ic + sl * x); sv = sv.set_index("volume_id")          # boundary measure: residual log SA/V after log volume (fit on all 32)
vol = lambda ids: float(sv.loc[list(ids)].vol.mean()); res = lambda ids: float(sv.loc[list(ids)].res.mean())

rows = []
for f in sorted(glob.glob(f"{E1_DIR}/e1_*.json")):
    r = load(f); d = json.load(open(f))
    if d["selection_seed"] == d["training_seed"]: rows.append(dict(r, group="cloud"))        # diagonal runs of the new E1 subsets
for f in sorted(glob.glob(f"{EXT_DIR}/*.json")): rows.append(dict(load(f), group="cloud"))
for s in (42, 123, 7): rows.append(dict(load(f"{V2_DIR}/v2_N05_random_attentionunet_seed{s}.json"), group="cloud"))
for f in sorted(glob.glob(f"{E4_DIR}/e4a_*.json")): rows.append(dict(load(f), group="E4a"))
for f in sorted(glob.glob(f"{E4_DIR}/e4b_*.json")): rows.append(dict(load(f), group="E4b"))
D = pd.DataFrame(rows)
assert (D.group == "cloud").sum() == 19 and (D.group == "E4a").sum() == 6 and (D.group == "E4b").sum() == 6
D["vol"] = D.ids.apply(vol); D["res"] = D.ids.apply(res)
e2 = [load(f) for f in sorted(glob.glob(f"{E1_DIR}/e2_*.json"))]; assert len(e2) == 3
E2 = dict(best=np.mean([r["best"] for r in e2]), last5=np.mean([r["last5"] for r in e2]), vol=vol(e2[0]["ids"]), res=res(e2[0]["ids"]))

# ---------------- comparisons C-1, C-2, C-3 ----------------
def diff_test(a, b):
    obs = a.mean() - b.mean()
    boots = np.array([RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(NBOOT)])
    pool = np.concatenate([a, b]); n = len(a); cnt = 0
    for _ in range(NPERM):
        p = RNG.permutation(pool); cnt += abs(p[:n].mean() - p[n:].mean()) >= abs(obs) - 1e-12
    return obs, np.percentile(boots, [2.5, 97.5]), (cnt + 1) / (NPERM + 1)

# ---------------- joint regression (31 subsets), OLS + HC3 ----------------
def ols_hc3(X, yv):
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ yv
    e = yv - X @ b; h = np.einsum("ij,jk,ik->i", X, XtXi, X); w = (e / (1 - h)) ** 2
    V = XtXi @ (X.T * w) @ X @ XtXi; return b, np.sqrt(np.diag(V)), n - k

def regression(df, col):
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    X = np.column_stack([np.ones(len(df)), z(df.vol.values), z(df.res.values)]); b, se, dof = ols_hc3(X, df[col].values)
    return dict(b=b, se=se, dof=dof, n=len(df), ci_norm=[(b[i] - 1.96 * se[i], b[i] + 1.96 * se[i]) for i in range(3)],
                ci_t=[(b[i] - stats.t.ppf(.975, dof) * se[i], b[i] + stats.t.ppf(.975, dof) * se[i]) for i in range(3)])

out = {}
for col in ("best", "last5"):
    c = D[D.group == "cloud"][col].values; a = D[D.group == "E4a"][col].values; b = D[D.group == "E4b"][col].values
    o = dict(cloud_mean=c.mean(), cloud_sd=c.std(ddof=1), e4a_mean=a.mean(), e4b_mean=b.mean(), e2=E2[col])
    o["C1"], o["C2"], o["C3"] = diff_test(b, c), diff_test(a, c), diff_test(b, a)
    o["reg"] = regression(D, col)
    D31 = pd.concat([D, pd.DataFrame([dict(group="E2", vol=E2["vol"], res=E2["res"], best=E2["best"], last5=E2["last5"])])], ignore_index=True)
    o["reg_withE2"] = regression(D31, col)
    out[col] = o

def sig_pos(t): return t[0] > 0 and t[1][0] > 0
def decisions(col):
    o = out[col]; c1 = sig_pos(o["C1"]); c2 = sig_pos(o["C2"]); r = o["reg"]; bres = r["b"][2] > 0 and r["ci_norm"][2][0] > 0
    gap = o["e2"] - o["cloud_mean"]
    return dict(C1_pos_sig=c1, bres_pos_sig=bres, D1=c1 and bres, C2_pos_sig=c2, D2=c2, D2_accounts=c2 and o["C2"][0] >= 0.5 * gap, gap_E2=gap)

if __name__ == "__main__":
    f = lambda v, d=2: f"{v:.{d}f}"
    print("SUBSETS (single run each; DSC points)")
    print(D[D.group != "cloud"].assign(sub=lambda t: t.run_id.str.extract(r"sel(\d+)")[0])[["group", "sub", "best", "last5", "vol", "res", "epochs"]].round(3).to_string(index=False))
    print(f"\ncloud (19 diagonal random-subset runs): best mean {f(out['best']['cloud_mean'])} SD {f(out['best']['cloud_sd'])} | last5 mean {f(out['last5']['cloud_mean'])} SD {f(out['last5']['cloud_sd'])}")
    print(f"E2 (mean of its 3 runs, excluded from the confirmatory analysis): best {f(E2['best'])} | last5 {f(E2['last5'])} | volume {f(E2['vol'],0)} mL | boundary measure {f(E2['res'],3)}")
    for col in ("best", "last5"):
        o = out[col]; print(f"\n=== [{col}] group means: cloud {f(o['cloud_mean'])} | E4a {f(o['e4a_mean'])} | E4b {f(o['e4b_mean'])} ===")
        for nm, key in (("C-1 E4b - cloud", "C1"), ("C-2 E4a - cloud", "C2"), ("C-3 E4b - E4a", "C3")):
            d, ci, p = o[key]; print(f"  {nm:16s} diff {d:+.2f}  95% bootstrap CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  permutation p {p:.3f}")
        for lab, key in (("regression, 31 subsets (E2 excluded)", "reg"), ("sensitivity: + E2 as 32nd point", "reg_withE2")):
            r = o[key]; print(f"  {lab}: n={r['n']}")
            for i, nm in ((1, "b_vol (per SD of volume)"), (2, "b_res (per SD of boundary measure)")):
                print(f"     {nm:34s} {r['b'][i]:+.2f}  HC3 SE {r['se'][i]:.2f}  95% CI normal [{r['ci_norm'][i][0]:+.2f}, {r['ci_norm'][i][1]:+.2f}] | t [{r['ci_t'][i][0]:+.2f}, {r['ci_t'][i][1]:+.2f}]")
    print("\n=== Decision rules (addendum 3, C3) ===")
    for col in ("best", "last5"):
        d = decisions(col); print(f"[{col}] C-1>0 & CI excludes 0: {d['C1_pos_sig']} | b_res>0 & CI excludes 0: {d['bres_pos_sig']} -> D1 (boundary measure supported): {d['D1']}")
        print(f"       C-2>0 & CI excludes 0: {d['C2_pos_sig']} -> D2 (volume contributes): {d['D2']} | accounts for E2's advantage (C-2 lower-bound rule, gap {d['gap_E2']:.2f}): {d['D2_accounts']}")

    # ---------------- exploratory, NOT pre-registered (descriptive) ----------------
    print("\n=== Exploratory (not pre-registered): spread and rank of E2 among all subsets ===")
    for col in ("best", "last5"):
        allv = {g: D[D.group == g][col].values for g in ("cloud", "E4a", "E4b")}
        print(f"[{col}] SD of single-run DSC: cloud {allv['cloud'].std(ddof=1):.2f} | E4a {allv['E4a'].std(ddof=1):.2f} | E4b {allv['E4b'].std(ddof=1):.2f} ; range cloud [{allv['cloud'].min():.1f}, {allv['cloud'].max():.1f}] E4a [{allv['E4a'].min():.1f}, {allv['E4a'].max():.1f}] E4b [{allv['E4b'].min():.1f}, {allv['E4b'].max():.1f}] ; all 31 subsets [{D[col].min():.1f}, {D[col].max():.1f}]")
        rank = 1 + int((D[col].values > E2[col]).sum())
        print(f"       E2 ({E2[col]:.2f}) would rank {rank} of 32 among all 31 single-run subsets plus itself; subsets above E2: {sorted(D[D[col] > E2[col]].run_id.str.extract(r'(e4[ab]_N05_[a-z]+|e[0-9]_N05_[a-z]+)')[0].tolist())}")
        print(f"       volume-matched random subsets (E4a) above E2: {int((allv['E4a'] > E2[col]).sum())} of 6 ; E4b above E2: {int((allv['E4b'] > E2[col]).sum())} of 6")
    print("E4b upper end of the C-1 interval (best-epoch) vs the apparent advantage of E2 over the cloud: %+.2f vs %+.2f" % (out["best"]["C1"][1][1], out["best"]["e2"] - out["best"]["cloud_mean"]))
    # descriptive
    c = D[D.group == "cloud"]
    print("\ncorrelations in the cloud (19): DSC-vol %.2f, DSC-res %.2f; E4a DSC-vol/E4b DSC-res are not tested (6 points)" % (c.best.corr(c.vol), c.best.corr(c.res)))
