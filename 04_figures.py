# 04_figures.py -- run order: 4  (requires 03_run_analysis.py outputs; refits final model to compute PDP)
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import importlib.util, sys
R = json.load(open("out/results.json")); curves = json.load(open("out/curves.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
GR = ["#000000", "#555555", "#888888", "#bbbbbb"]
# ---- refit final model (same code path/params as 03) to get partial dependence on non-missing values ----
src = open("03_run_analysis.py").read().split("# ======================= PRIMARY EXPERIMENT")[0]
ns = {}; exec(compile(src, "03_head", "exec"), ns)
hgb, load, split, ALL = ns["hgb"], ns["load"], ns["split"], ns["ALL"]
d = load(30); tr, te = split(d)
best = [m for m in R["models"] if m["model"].startswith("HistGradient")][0]; bp = json.loads(best["best_params"])
final = hgb(ALL, learning_rate=bp["clf__learning_rate"], max_leaf_nodes=bp["clf__max_leaf_nodes"]).fit(tr[ALL], tr.dropout)
auc = roc_auc_score(te.dropout, final.predict_proba(te[ALL])[:, 1]); print("refit test AUC", auc, "reported", best["AUC"])
assert abs(auc - best["AUC"]) < 1e-9
pdp = {}
for f in ["mean_score", "n_missed", "mean_days_late", "studied_credits"]:
    x = te[f].dropna(); grid = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 15))); ys = []
    for v in grid:
        X = te[ALL].copy(); X[f] = v; ys.append(float(final.predict_proba(X)[:, 1].mean()))
    pdp[f] = {"x": grid.tolist(), "y": ys, "n_nonmissing": int(x.notna().sum()), "n_total": int(len(te))}
R["pdp_fixed"] = pdp; json.dump(R, open("out/results.json", "w"), indent=1, default=str)
# Fig 1: AUC by cutoff
cs = pd.DataFrame(R["cutoffs"]); fig, ax = plt.subplots(figsize=(5.5, 3.6))
for nm, mk, ls, c in [("HistGradientBoosting (proposed)", "o", "-", GR[0]), ("Logistic regression", "s", "--", GR[1]), ("Rule: no VLE activity", "^", ":", GR[2])]:
    s = cs[cs.model == nm]; ax.errorbar(s.cutoff_day, s.AUC, yerr=[s.AUC - s.AUC_lo, s.AUC_hi - s.AUC], marker=mk, ls=ls, color=c, capsize=2, label=nm.replace(" (proposed)", ""))
ax.set_xlabel("Cutoff day (days from module start)"); ax.set_ylabel("Test AUC (2014J)"); ax.set_ylim(0.55, 0.77); ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("out/figs/fig1_auc_by_cutoff.png", dpi=300); plt.close()
# Fig 2: ROC / PR / calibration
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4)); sty = dict(zip(curves.keys(), [("-", GR[3]), ("-.", GR[2]), ("--", GR[1]), (":", GR[1]), ("-", GR[0])]))
for nm, c in curves.items():
    ls, col = sty[nm]; axs[0].plot(c["fpr"], c["tpr"], ls=ls, color=col, label=nm.replace(" (proposed)", "")); axs[1].plot(c["rec"], c["prec"], ls=ls, color=col)
axs[0].plot([0, 1], [0, 1], color="#cccccc", lw=0.8); axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].legend(frameon=False, fontsize=7); axs[0].set_title("(a) ROC", fontsize=9)
axs[1].axhline(R["primary"]["dropout_test"], color="#cccccc", lw=0.8); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("(b) Precision-recall", fontsize=9)
cal = R["calibration"]; axs[2].plot([0, 0.5], [0, 0.5], color="#cccccc", lw=0.8); axs[2].plot(cal["mean_pred"], cal["frac_pos"], "o-", color=GR[0]); axs[2].set_xlabel("Mean predicted probability"); axs[2].set_ylabel("Observed dropout rate"); axs[2].set_title("(c) Calibration (HGB)", fontsize=9)
plt.tight_layout(); plt.savefig("out/figs/fig2_roc_pr_calibration.png", dpi=300); plt.close()
# Fig 3: importance
pi = pd.read_csv("out/table_perm_importance.csv").head(12)[::-1]; gi = pd.read_csv("out/table_group_importance.csv")[::-1]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
axs[0].barh(pi.feature, pi.auc_drop_mean, xerr=pi.auc_drop_sd, color=GR[1], ecolor="k", capsize=2); axs[0].set_xlabel("Mean AUC decrease after permutation"); axs[0].set_title("(a) Top 12 features", fontsize=9)
axs[1].barh(gi.group, gi.auc_drop_mean, xerr=gi.auc_drop_sd, color=GR[2], ecolor="k", capsize=2); axs[1].set_xlabel("Mean AUC decrease (group permutation)"); axs[1].set_title("(b) Feature groups", fontsize=9)
plt.tight_layout(); plt.savefig("out/figs/fig3_importance.png", dpi=300); plt.close()
# Fig 4: PDP
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
for ax, (f, v) in zip(axs, pdp.items()):
    ax.plot(v["x"], v["y"], "o-", color=GR[0], ms=3); ax.set_xlabel(f); ax.set_title(f"n non-missing = {v['n_nonmissing']}", fontsize=7)
axs[0].set_ylabel("Mean predicted dropout probability"); plt.tight_layout(); plt.savefig("out/figs/fig4_pdp.png", dpi=300); plt.close()
# Fig 5: fairness gaps under three conditions
att = ["gender", "disability", "age2", "imd3"]; lab = ["Gender", "Disability", "Age (<35 / 35+)", "IMD (3 bands)"]
conds = [("All features", [R["fairness_all"][a]["gaps"] for a in att]), ("Unaware (no protected features)", [R["fairness_unaware"][a]["gaps"] for a in att]),
         ("Group thresholds (TPR-equalised)", [R["mitigation"]["by_attribute"][a]["gaps"] for a in att])]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); w = 0.26
for j, key in enumerate(["TPR_gap", "FPR_gap"]):
    for i, (cn, gl) in enumerate(conds):
        axs[j].bar(np.arange(4) + (i - 1) * w, [g[key] for g in gl], w, color=GR[i], label=cn)
    axs[j].set_xticks(range(4)); axs[j].set_xticklabels(lab, fontsize=8); axs[j].set_ylabel(("TPR" if j == 0 else "FPR") + " gap (max - min)")
    axs[j].set_title(("(a) Equal-opportunity gap" if j == 0 else "(b) False-positive-rate gap"), fontsize=9)
axs[0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig("out/figs/fig5_fairness_gaps.png", dpi=300); plt.close()
# Fig 6: leave-one-module-out
lo = pd.read_csv("out/table_lomo.csv"); fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(lo.held_out_module, lo.AUC, color=GR[1]); ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.set_ylim(0.45, 0.75); ax.set_xlabel("Held-out module"); ax.set_ylabel("AUC (day 30)")
for i, (a_, n_) in enumerate(zip(lo.AUC, lo.n_test)): ax.text(i, a_ + 0.005, f"{a_:.2f}", ha="center", fontsize=7)
plt.tight_layout(); plt.savefig("out/figs/fig6_lomo.png", dpi=300); plt.close()
print(json.dumps(pdp)[:900]); print(R["fairness_all"]["highest_education"]["groups"]); print({k:(v["n"],v["n_dropout"],round(v["TPR"],2),round(v["FPR"],2)) for k,v in R["fairness_all"]["region"]["groups"].items()})
