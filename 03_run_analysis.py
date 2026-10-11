# 03_run_analysis.py -- run order: 3   (python 3.x, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4)
# Primary experiment: cutoff day 30; temporal hold-out (train 2013B/2013J/2014B, test 2014J).
import pandas as pd, numpy as np, json, time, sys, warnings, os
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.inspection import partial_dependence
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)
PRIMARY = 30; CUTOFFS = [0, 14, 30, 60, 90, 120]; B = 1000
TRAIN_PRES = ["2013B", "2013J", "2014B"]; TEST_PRES = ["2014J"]
os.makedirs("out/figs", exist_ok=True)
R = {"versions": {"python": sys.version.split()[0]}}
import sklearn, matplotlib
R["versions"].update({"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__})
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def save():
    json.dump(R, open("out/results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ------------------------- feature groups -------------------------
IMD = {"0-10%":0,"10-20":1,"10-20%":1,"20-30%":2,"30-40%":3,"40-50%":4,"50-60%":5,"60-70%":6,"70-80%":7,"80-90%":8,"90-100%":9}
PROT = ["gender", "region", "imd_decile", "age_band", "disability"]
EDU = ["highest_education"]
COURSE = ["code_module", "term", "num_of_prev_attempts", "studied_credits"]
REG = ["date_registration"]
VLE = ["clicks_total","active_days","n_activity_types","first_click_day","clicks_last7","clicks_prev7","clicks_prestart",
       "clk_forumng","clk_homepage","clk_other","clk_oucontent","clk_page","clk_quiz","clk_resource","clk_subpage","clk_url",
       "clicks_trend","clicks_per_active_day","days_since_last_click","no_activity"]
ASSESS = ["n_due","n_submitted","mean_score","mean_days_late","weighted_score_sum","n_submitted_due","n_missed"]
CAT = ["gender","region","highest_education","age_band","disability","code_module","term"]
NOLOG = ["first_click_day","days_since_last_click","no_activity","date_registration","imd_decile","num_of_prev_attempts",
         "studied_credits","n_activity_types","n_due","mean_score","mean_days_late"]
GROUPS = {"Demographic/protected": PROT, "Prior education": EDU, "Course context": COURSE, "Registration": REG,
          "VLE clickstream": VLE, "Assessment": ASSESS}
ALL = PROT + EDU + COURSE + REG + VLE + ASSESS
UNAWARE = [c for c in ALL if c not in PROT]

def load(t):
    d = pd.read_csv(f"data/features_t{t}.csv")
    d["imd_decile"] = d.imd_band.map(IMD)
    return d
def signed_log(x): return np.sign(x) * np.log1p(np.abs(x))

def make_pre(cols, kind):
    cat = [c for c in cols if c in CAT]; num = [c for c in cols if c not in CAT]
    tf = []
    if cat: tf.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if kind == "hgb":
        tf.append(("num", "passthrough", num))
    elif kind == "tree":
        tf.append(("num", SimpleImputer(strategy="median", add_indicator=True), num))
    else:  # scaled models: signed-log on skewed counts, impute, standardise
        sk = [c for c in num if c not in NOLOG]; ot = [c for c in num if c in NOLOG]
        if sk: tf.append(("skew", Pipeline([("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
                      ("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), sk))
        if ot: tf.append(("oth", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), ot))
    return ColumnTransformer(tf, remainder="drop")

def hgb(cols, **kw):
    p = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, max_iter=300,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=SEED); p.update(kw)
    return Pipeline([("pre", make_pre(cols, "hgb")), ("clf", HistGradientBoostingClassifier(**p))])

def model_specs(cols):
    return {
     "Logistic regression": (Pipeline([("pre", make_pre(cols, "scaled")), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
                             {"clf__C": [0.01, 0.1, 1.0]}),
     "Decision tree": (Pipeline([("pre", make_pre(cols, "tree")), ("clf", DecisionTreeClassifier(random_state=SEED))]),
                       {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [20, 50]}),
     "Random forest": (Pipeline([("pre", make_pre(cols, "tree")), ("clf", RandomForestClassifier(n_estimators=150, max_features="sqrt", random_state=SEED, n_jobs=1))]),
                       {"clf__max_depth": [10, 16], "clf__min_samples_leaf": [5, 20]}),
     "MLP (64-32)": (Pipeline([("pre", make_pre(cols, "scaled")), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=150, random_state=SEED))]),
                     {"clf__alpha": [1e-3, 1e-2]}),
     "HistGradientBoosting (proposed)": (hgb(cols), {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]}),
    }

def cv(): return GroupKFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------- metrics -------------------------
def best_threshold(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    f = [prf(y, (p >= t).astype(int))[2] for t in ts]
    return float(ts[int(np.argmax(f))])
def prf(y, yhat):
    tp = ((yhat == 1) & (y == 1)).sum(); fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1
def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0
    for i in range(bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
def topk(y, p, frac=0.2):
    k = int(round(frac * len(y))); o = np.argsort(-p)[:k]
    return float(y[o].mean()), float(y[o].sum() / y.sum())
def metrics(y, p, thr):
    yhat = (p >= thr).astype(int); pr, rc, f1 = prf(y, yhat); pk, rk = topk(y, p)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p), "Precision": pr, "Recall": rc, "F1": f1,
            "Accuracy": float((yhat == y).mean()), "Brier": brier_score_loss(y, p), "ECE": ece(y, p),
            "Prec@top20": pk, "Recall@top20": rk, "threshold": thr}
IDX = None
def boot_idx(n):
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))
def boot_ci(y, p, thr, idx):
    out = {"AUC": [], "PR_AUC": [], "F1": []}
    for ix in idx:
        yy, pp = y[ix], p[ix]
        if yy.min() == yy.max(): continue
        out["AUC"].append(roc_auc_score(yy, pp)); out["PR_AUC"].append(average_precision_score(yy, pp))
        out["F1"].append(prf(yy, (pp >= thr).astype(int))[2])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
def boot_auc_diff(y, p1, p2, idx):
    d = np.array([roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p2[ix]) for ix in idx if y[ix].min() != y[ix].max()])
    pval = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_boot": pval}

# ------------------------- split -------------------------
def split(d):
    tr = d[d.code_presentation.isin(TRAIN_PRES)].reset_index(drop=True)
    te = d[d.code_presentation.isin(TEST_PRES)].reset_index(drop=True)
    return tr, te

def tune_and_predict(name, est, grid, tr, te, cols):
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv(), n_jobs=1, refit=True)
    gs.fit(tr[cols], tr.dropout, groups=tr.id_student)
    best = gs.best_estimator_
    oof = cross_val_predict(clone(best), tr[cols], tr.dropout, cv=cv(), groups=tr.id_student, method="predict_proba")[:, 1]
    thr = best_threshold(tr.dropout.values, oof)
    p = best.predict_proba(te[cols])[:, 1]
    return {"best_params": {k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in gs.best_params_.items()},
            "cv_auc_mean": float(gs.best_score_),
            "cv_auc_sd": float(gs.cv_results_["std_test_score"][gs.best_index_]), "model": best, "oof": oof, "thr": thr, "p": p}

# ======================= PRIMARY EXPERIMENT =======================
d = load(PRIMARY); tr, te = split(d)
ytr, yte = tr.dropout.values, te.dropout.values
IDX = boot_idx(len(te))
R["primary"] = {"cutoff": PRIMARY, "n_train": len(tr), "n_test": len(te), "dropout_train": float(ytr.mean()), "dropout_test": float(yte.mean()),
                "n_dropout_train": int(ytr.sum()), "n_dropout_test": int(yte.sum()),
                "unique_students_train": int(tr.id_student.nunique()), "unique_students_test": int(te.id_student.nunique()),
                "test_students_also_in_train": int(te.id_student.isin(tr.id_student).sum()),
                "train_presentations": TRAIN_PRES, "test_presentations": TEST_PRES, "bootstrap_reps": B,
                "n_features_raw": len(ALL)}
log("primary", R["primary"])

# table 1: cohort characteristics
rows = []
for c in ["gender", "disability", "age_band", "highest_education", "code_module"]:
    for lv in sorted(d[c].dropna().unique()):
        rows.append({"variable": c, "level": lv, "train_n": int((tr[c] == lv).sum()), "train_pct": float((tr[c] == lv).mean() * 100),
                     "train_dropout_pct": float(tr.loc[tr[c] == lv, "dropout"].mean() * 100),
                     "test_n": int((te[c] == lv).sum()), "test_pct": float((te[c] == lv).mean() * 100),
                     "test_dropout_pct": float(te.loc[te[c] == lv, "dropout"].mean() * 100)})
pd.DataFrame(rows).to_csv("out/table_cohort.csv", index=False)

# ---- models ----
specs = model_specs(ALL); res = {}
for nm, (est, grid) in specs.items():
    res[nm] = tune_and_predict(nm, est, grid, tr, te, ALL); log("tuned", nm, res[nm]["best_params"], round(res[nm]["cv_auc_mean"], 4))
PROPOSED = "HistGradientBoosting (proposed)"
# simple baselines
rule_score = -te.clicks_total.values.astype(float)           # fewer clicks -> higher risk
rule_thr = None
mrows = []; boots = {}
for nm, r in res.items():
    m = metrics(yte, r["p"], r["thr"]); ci = boot_ci(yte, r["p"], r["thr"], IDX); boots[nm] = ci
    mrows.append({"model": nm, **m, **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()},
                  "cv_auc": r["cv_auc_mean"], "cv_auc_sd": r["cv_auc_sd"], "best_params": json.dumps(r["best_params"])})
# no-activity rule
yh = te.no_activity.values; pr_, rc_, f_ = prf(yte, yh)
ci = boot_ci(yte, (-np.log1p(te.clicks_total.values)), 0.5, IDX)
mrows.append({"model": "Rule: no VLE activity by day 30", "AUC": roc_auc_score(yte, rule_score), "PR_AUC": average_precision_score(yte, rule_score),
              "Precision": pr_, "Recall": rc_, "F1": f_, "Accuracy": float((yh == yte).mean()), "AUC_lo": ci["AUC"][0], "AUC_hi": ci["AUC"][1],
              "PR_AUC_lo": ci["PR_AUC"][0], "PR_AUC_hi": ci["PR_AUC"][1]})
mrows.append({"model": "Prevalence (all-positive)", "AUC": 0.5, "Precision": float(yte.mean()), "Recall": 1.0,
              "F1": 2 * yte.mean() / (1 + yte.mean()), "Accuracy": float(yte.mean())})
mt = pd.DataFrame(mrows); mt.to_csv("out/table_models.csv", index=False)
R["models"] = mt.to_dict("records")
R["auc_diff_vs_proposed"] = {nm: boot_auc_diff(yte, res[PROPOSED]["p"], res[nm]["p"], IDX) for nm in res if nm != PROPOSED}
R["auc_diff_vs_proposed"]["Rule: no VLE activity"] = boot_auc_diff(yte, res[PROPOSED]["p"], rule_score, IDX)
log("models done"); save()
pd.DataFrame({"code_module": te.code_module, "code_presentation": te.code_presentation, "id_student": te.id_student, "y": yte,
              **{nm: r["p"] for nm, r in res.items()}}).to_csv("out/test_predictions_t30.csv", index=False)

# sensitivity: test excluding students seen in training
mask = ~te.id_student.isin(tr.id_student).values
R["sens_excl_overlap"] = {"n": int(mask.sum()), "dropout": float(yte[mask].mean()),
    **{nm: float(roc_auc_score(yte[mask], res[nm]["p"][mask])) for nm in res}}
# seed stability of the proposed model
bp = res[PROPOSED]["best_params"]; aucs = []
for s in range(5):
    m_ = hgb(ALL, random_state=100 + s, learning_rate=bp["clf__learning_rate"], max_leaf_nodes=bp["clf__max_leaf_nodes"]).fit(tr[ALL], ytr)
    aucs.append(roc_auc_score(yte, m_.predict_proba(te[ALL])[:, 1]))
R["seed_stability_auc"] = {"values": aucs, "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1))}
log("sens/seeds", R["sens_excl_overlap"], R["seed_stability_auc"]); save()

# ---- ablation (proposed model's tuned params) ----
hp = dict(learning_rate=bp["clf__learning_rate"], max_leaf_nodes=bp["clf__max_leaf_nodes"])
def eval_set(cols):
    m_ = hgb(cols, **hp).fit(tr[cols], ytr); p_ = m_.predict_proba(te[cols])[:, 1]
    oof_ = cross_val_predict(hgb(cols, **hp), tr[cols], ytr, cv=cv(), groups=tr.id_student, method="predict_proba")[:, 1]
    thr_ = best_threshold(ytr, oof_); mm = metrics(yte, p_, thr_); ci_ = boot_ci(yte, p_, thr_, IDX)
    return {**mm, "AUC_lo": ci_["AUC"][0], "AUC_hi": ci_["AUC"][1], "PR_AUC_lo": ci_["PR_AUC"][0], "PR_AUC_hi": ci_["PR_AUC"][1], "n_features": len(cols)}
static = PROT + EDU + COURSE + REG
abl = {"Static only (demographic, education, course, registration)": static, "VLE clickstream only": VLE, "Assessment only": ASSESS,
       "Static + VLE": static + VLE, "Static + Assessment": static + ASSESS, "VLE + Assessment": VLE + ASSESS,
       "All features": ALL, "All minus protected attributes (unaware)": UNAWARE}
for gname, gcols in GROUPS.items():
    abl[f"All minus {gname}"] = [c for c in ALL if c not in gcols]
ab = []
for k, cols in abl.items():
    ab.append({"feature_set": k, **eval_set(cols)}); log("ablation", k, round(ab[-1]["AUC"], 4))
pd.DataFrame(ab).to_csv("out/table_ablation.csv", index=False); R["ablation"] = ab; save()

# ---- explainability ----
final = res[PROPOSED]["model"]; p_te = res[PROPOSED]["p"]; thr = res[PROPOSED]["thr"]
rng = np.random.default_rng(SEED)
base_auc = roc_auc_score(yte, p_te)
def perm_drop(cols, reps=10):
    ds = []
    for _ in range(reps):
        X = te[ALL].copy(); perm = rng.permutation(len(X))
        X[cols] = X[cols].values[perm]
        ds.append(base_auc - roc_auc_score(yte, final.predict_proba(X)[:, 1]))
    return float(np.mean(ds)), float(np.std(ds, ddof=1))
pi = []
for c in ALL:
    m_, s_ = perm_drop([c]); pi.append({"feature": c, "auc_drop_mean": m_, "auc_drop_sd": s_})
pi = pd.DataFrame(pi).sort_values("auc_drop_mean", ascending=False); pi.to_csv("out/table_perm_importance.csv", index=False)
gi = []
for g, cols in GROUPS.items():
    m_, s_ = perm_drop(cols); gi.append({"group": g, "auc_drop_mean": m_, "auc_drop_sd": s_})
gi = pd.DataFrame(gi).sort_values("auc_drop_mean", ascending=False); gi.to_csv("out/table_group_importance.csv", index=False)
R["perm_importance_top"] = pi.head(15).to_dict("records"); R["group_importance"] = gi.to_dict("records")
log("perm importance done"); save()

# logistic-regression odds ratios (standardised features)
lr = res["Logistic regression"]["model"]; names = lr.named_steps["pre"].get_feature_names_out()
coef = lr.named_steps["clf"].coef_[0]
orr = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "odds_ratio_per_SD": np.exp(coef), "coef": coef})
orr["abs"] = orr.coef.abs(); orr = orr.sort_values("abs", ascending=False).drop(columns="abs"); orr.to_csv("out/table_lr_odds.csv", index=False)
R["lr_odds_top"] = orr.head(12).to_dict("records")

# partial dependence on top numeric features
topnum = [f for f in pi.feature if f not in CAT][:4]
pdp = {}
sample = te[ALL].sample(3000, random_state=SEED)
for f in topnum:
    r_ = partial_dependence(final, sample, [f], grid_resolution=30, kind="average")
    pdp[f] = {"x": [float(v) for v in r_["grid_values"][0]], "y": [float(v) for v in r_["average"][0]]}
R["pdp"] = pdp

# global surrogate tree (depth 3) fit to the proposed model's class predictions
Xtr_t = final.named_steps["pre"].transform(tr[ALL]); Xte_t = final.named_steps["pre"].transform(te[ALL])
fn = [n.split("__", 1)[1] for n in final.named_steps["pre"].get_feature_names_out()]
yh_tr = (final.predict_proba(tr[ALL])[:, 1] >= thr).astype(int); yh_te = (p_te >= thr).astype(int)
sur = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED).fit(Xtr_t, yh_tr)
R["surrogate"] = {"fidelity_test": float((sur.predict(Xte_t) == yh_te).mean()), "auc_vs_true_labels": float(roc_auc_score(yte, sur.predict_proba(Xte_t)[:, 1])),
                  "rules": export_text(sur, feature_names=fn, decimals=1)}
log("surrogate", R["surrogate"]["fidelity_test"]); save()

# ---- fairness ----
te["age2"] = np.where(te.age_band == "0-35", "0-35", "35+"); tr["age2"] = np.where(tr.age_band == "0-35", "0-35", "35+")
def imd3(x): return np.where(x.imd_decile.isna(), "Unknown", np.where(x.imd_decile <= 2, "IMD 0-30% (most deprived)", np.where(x.imd_decile <= 6, "IMD 30-70%", "IMD 70-100% (least deprived)")))
te["imd3"] = imd3(te); tr["imd3"] = imd3(tr)
FA = {"gender": "gender", "disability": "disability", "age2": "age2", "imd3": "imd3", "highest_education": "highest_education", "region": "region"}
def gstats(y, yhat, p, g):
    out = {}
    for lv in np.unique(g):
        m = g == lv; yy, hh = y[m], yhat[m]
        pos, neg = (yy == 1).sum(), (yy == 0).sum()
        out[lv] = {"n": int(m.sum()), "n_dropout": int(pos), "base_rate": float(yy.mean()), "selection": float(hh.mean()),
                   "TPR": float(hh[yy == 1].mean()) if pos else np.nan, "FPR": float(hh[yy == 0].mean()) if neg else np.nan,
                   "PPV": float(yy[hh == 1].mean()) if hh.sum() else np.nan,
                   "AUC": float(roc_auc_score(yy, p[m])) if pos and neg else np.nan}
    return out
def gaps(st, minn=50, minpos=10):
    ok = {k: v for k, v in st.items() if v["n"] >= minn and v["n_dropout"] >= minpos and k != "Unknown"}
    f = lambda key: (max(v[key] for v in ok.values()) - min(v[key] for v in ok.values())) if len(ok) > 1 else np.nan
    sel = [v["selection"] for v in ok.values()]
    return {"DP_gap": f("selection"), "TPR_gap": f("TPR"), "FPR_gap": f("FPR"), "EO_gap": max(f("TPR"), f("FPR")),
            "DI_ratio": (min(sel) / max(sel)) if len(ok) > 1 and max(sel) > 0 else np.nan, "groups_included": list(ok.keys())}
def fair_report(p, thr, label, boot=True):
    yhat = (p >= thr).astype(int); out = {}
    for a, col in FA.items():
        g = te[col].astype(str).values; st = gstats(yte, yhat, p, g); gp = gaps(st); rec = {"groups": st, "gaps": gp}
        if boot and a in ("gender", "disability", "age2", "imd3"):
            bs = {"DP_gap": [], "TPR_gap": [], "FPR_gap": []}
            for ix in IDX[:500]:
                s2 = gstats(yte[ix], yhat[ix], p[ix], g[ix]); g2 = gaps(s2)
                for k in bs: bs[k].append(g2[k])
            rec["gap_ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
        out[a] = rec
    return out
R["fairness_all"] = fair_report(p_te, thr, "all")
log("fairness (all features) done"); save()
# unaware model
mu = hgb(UNAWARE, **hp).fit(tr[UNAWARE], ytr); pu = mu.predict_proba(te[UNAWARE])[:, 1]
oofu = cross_val_predict(hgb(UNAWARE, **hp), tr[UNAWARE], ytr, cv=cv(), groups=tr.id_student, method="predict_proba")[:, 1]
thr_u = best_threshold(ytr, oofu)
R["fairness_unaware"] = fair_report(pu, thr_u, "unaware"); R["unaware_metrics"] = metrics(yte, pu, thr_u)
log("fairness (unaware) done"); save()
# post-processing: group-specific thresholds equalising TPR on out-of-fold training predictions
oof = res[PROPOSED]["oof"]
tpr_target = float(((oof >= thr) & (ytr == 1)).sum() / (ytr == 1).sum())
mit = {}
for a in ["gender", "disability", "age2", "imd3"]:
    gtr = tr[a].astype(str).values; gte = te[a].astype(str).values; ths = {}
    for lv in np.unique(gtr):
        m = (gtr == lv) & (ytr == 1)
        ths[lv] = float(np.quantile(oof[m], 1 - tpr_target)) if m.sum() >= 30 else thr
    yhat2 = np.array([p_te[i] >= ths.get(gte[i], thr) for i in range(len(te))]).astype(int)
    st = gstats(yte, yhat2, p_te, gte); gp = gaps(st); pr_, rc_, f_ = prf(yte, yhat2)
    mit[a] = {"thresholds": ths, "gaps": gp, "groups": st, "Precision": pr_, "Recall": rc_, "F1": f_}
R["mitigation"] = {"tpr_target": tpr_target, "global_threshold": thr, "baseline_F1": float(R["models"][4]["F1"]) if "F1" in R["models"][4] else None, "by_attribute": mit}
log("mitigation done"); save()

# ---- error analysis ----
unreg = pd.read_csv("data/unreg_dates.csv")
te2 = te.merge(unreg, on=["code_module", "code_presentation", "id_student"], how="left")
te2["pred"] = yh_te; te2["p"] = p_te
te2["days_to_wd"] = te2.date_unregistration - PRIMARY
pos = te2[te2.dropout == 1].copy()
bins = [-np.inf, 14, 60, 120, np.inf]; labs = ["<=14 days after cutoff", "15-60", "61-120", ">120"]
pos["bucket"] = pd.cut(pos.days_to_wd, bins=bins, labels=labs); pos["bucket"] = pos.bucket.astype(object).where(pos.days_to_wd.notna(), "no unregistration date")
R["error_analysis"] = {
  "confusion": {"TP": int(((yh_te == 1) & (yte == 1)).sum()), "FP": int(((yh_te == 1) & (yte == 0)).sum()),
                "FN": int(((yh_te == 0) & (yte == 1)).sum()), "TN": int(((yh_te == 0) & (yte == 0)).sum())},
  "recall_by_withdrawal_timing": {str(k): {"n": int(len(g)), "recall": float(g.pred.mean())} for k, g in pos.groupby("bucket")},
  "recall_by_module": {k: {"n": int(len(g)), "recall": float(g.pred.mean())} for k, g in pos.groupby("code_module")},
  "precision_by_module": {k: float(g[g.pred == 1].dropout.mean()) if (g.pred == 1).sum() else None for k, g in te2.groupby("code_module")},
  "fp_final_result": te2[(te2.pred == 1) & (te2.dropout == 0)].final_result.value_counts().to_dict(),
  "tn_final_result": te2[(te2.pred == 0) & (te2.dropout == 0)].final_result.value_counts().to_dict(),
  "dropouts_tp_vs_fn_medians": {c: {"TP": float(pos[pos.pred == 1][c].median()), "FN": float(pos[pos.pred == 0][c].median())}
                                for c in ["clicks_total", "active_days", "n_submitted", "mean_score", "date_registration", "studied_credits"]},
  "dropouts_tp_vs_fn_no_activity_share": {"TP": float(pos[pos.pred == 1].no_activity.mean()), "FN": float(pos[pos.pred == 0].no_activity.mean())}}
log("error analysis done"); save()

# ---- calibration & curves (for figures) ----
fr, mp = calibration_curve(yte, p_te, n_bins=10, strategy="quantile")
R["calibration"] = {"frac_pos": fr.tolist(), "mean_pred": mp.tolist()}
curves = {}
for nm, r in res.items():
    f_, t_, _ = roc_curve(yte, r["p"]); pp_, rr_, _ = precision_recall_curve(yte, r["p"])
    curves[nm] = {"fpr": f_.tolist(), "tpr": t_.tolist(), "prec": pp_.tolist(), "rec": rr_.tolist()}
json.dump(curves, open("out/curves.json", "w"))
save()

# ======================= LEAVE-ONE-MODULE-OUT (generalisability) =======================
lomo = []
for mod in sorted(d.code_module.unique()):
    te_m = d[d.code_module == mod]; tr_m = d[(d.code_module != mod) & (~d.id_student.isin(te_m.id_student))]
    m_ = hgb(ALL, **hp).fit(tr_m[ALL], tr_m.dropout); p_ = m_.predict_proba(te_m[ALL])[:, 1]
    lomo.append({"held_out_module": mod, "n_test": len(te_m), "dropout_rate": float(te_m.dropout.mean()), "n_train": len(tr_m), "AUC": float(roc_auc_score(te_m.dropout, p_)),
                 "PR_AUC": float(average_precision_score(te_m.dropout, p_))})
pd.DataFrame(lomo).to_csv("out/table_lomo.csv", index=False); R["lomo"] = lomo; log("LOMO done"); save()

# ======================= CUTOFF SENSITIVITY =======================
cs = []
for t in CUTOFFS:
    dt = load(t); tr_t, te_t = split(dt); y_t = te_t.dropout.values; idx_t = boot_idx(len(te_t))
    for nm in ["Logistic regression", PROPOSED]:
        est, grid = model_specs(ALL)[nm]; r = tune_and_predict(nm, est, grid, tr_t, te_t, ALL)
        m = metrics(y_t, r["p"], r["thr"]); ci = boot_ci(y_t, r["p"], r["thr"], idx_t)
        cs.append({"cutoff_day": t, "model": nm, "n_train": len(tr_t), "n_test": len(te_t), "test_dropout_rate": float(y_t.mean()), **m,
                   "AUC_lo": ci["AUC"][0], "AUC_hi": ci["AUC"][1], "PR_AUC_lo": ci["PR_AUC"][0], "PR_AUC_hi": ci["PR_AUC"][1]})
    s_ = -np.log1p(te_t.clicks_total.values); ci = boot_ci(y_t, s_, 0.5, idx_t)
    cs.append({"cutoff_day": t, "model": "Rule: no VLE activity", "n_train": len(tr_t), "n_test": len(te_t), "test_dropout_rate": float(y_t.mean()),
               "AUC": roc_auc_score(y_t, s_), "PR_AUC": average_precision_score(y_t, s_), "AUC_lo": ci["AUC"][0], "AUC_hi": ci["AUC"][1]})
    log("cutoff", t, [round(x["AUC"], 4) for x in cs[-3:]])
    pd.DataFrame(cs).to_csv("out/table_cutoffs.csv", index=False); R["cutoffs"] = cs; save()
log("ALL DONE")
