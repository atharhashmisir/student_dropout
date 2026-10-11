# 02_build_features.py -- run order: 2
# Builds leakage-safe feature tables at fixed cutoff days t (days relative to module start).
# Only information dated <= t is used. Cohort = students still registered at t.
import pandas as pd, numpy as np, json
U = "/mnt/user-data/uploads/"
CUTOFFS = [0, 14, 30, 60, 90, 120]
KEY = ["code_module", "code_presentation", "id_student"]
GROUPS = {"homepage":"homepage","forumng":"forumng","oucontent":"oucontent","resource":"resource",
          "subpage":"subpage","quiz":"quiz","url":"url","page":"page"}   # all other types -> "other"

si = pd.read_csv(U+"studentInfo.csv"); sr = pd.read_csv(U+"studentRegistration.csv")
asm = pd.read_csv(U+"assessments.csv"); sa = pd.read_csv(U+"studentAssessment.csv")
crs = pd.read_csv(U+"courses.csv")
daily = pd.read_csv("data/vle_daily_by_type.csv")
daily["grp"] = daily.activity_type.map(GROUPS).fillna("other")

base = si.merge(sr, on=KEY, how="left").merge(crs, on=["code_module","code_presentation"], how="left")
base["dropout"] = (base.final_result == "Withdrawn").astype(int)
base["term"] = base.code_presentation.str[-1]            # B (Feb) / J (Oct) start
base["year"] = base.code_presentation.str[:4].astype(int)

# assessments (exclude Exam: dated late / missing dates)
a = sa.merge(asm, on="id_assessment", how="left")
a = a[a.assessment_type != "Exam"].merge(base[KEY].drop_duplicates(), on=["code_module","code_presentation","id_student"])
a["late"] = a.date_submitted - a.date
asm_ne = asm[asm.assessment_type != "Exam"]

summary = {}
for t in CUTOFFS:
    # cohort: registered by t (or unknown) and not unregistered at/before t
    coh = base[(base.date_unregistration.isna() | (base.date_unregistration > t)) &
               (base.date_registration.isna() | (base.date_registration <= t))].copy()
    # --- VLE features ---
    d = daily[daily.date <= t]
    g = d.groupby(KEY).agg(clicks_total=("clicks","sum"), active_days=("date","nunique"),
                           n_activity_types=("activity_type","nunique"),
                           first_click_day=("date","min"), last_click_day=("date","max"))
    last7 = d[d.date > t-7].groupby(KEY).clicks.sum().rename("clicks_last7")
    prev7 = d[(d.date > t-14) & (d.date <= t-7)].groupby(KEY).clicks.sum().rename("clicks_prev7")
    pre = d[d.date < 0].groupby(KEY).clicks.sum().rename("clicks_prestart")
    typ = d.groupby(KEY+["grp"]).clicks.sum().unstack(fill_value=0).add_prefix("clk_")
    v = g.join([last7, prev7, pre, typ], how="left").reset_index()
    coh = coh.merge(v, on=KEY, how="left")
    zero_cols = ["clicks_total","active_days","n_activity_types","clicks_last7","clicks_prev7","clicks_prestart"] + \
                [c for c in coh.columns if c.startswith("clk_")]
    for c in zero_cols: coh[c] = coh[c].fillna(0)
    coh["clicks_trend"] = coh.clicks_last7 - coh.clicks_prev7
    coh["clicks_per_active_day"] = np.where(coh.active_days > 0, coh.clicks_total / coh.active_days.replace(0, np.nan), 0)
    coh["days_since_last_click"] = np.where(coh.active_days > 0, t - coh.last_click_day, np.nan)
    coh["no_activity"] = (coh.clicks_total == 0).astype(int)
    coh = coh.drop(columns=["last_click_day"])
    # --- assessment features ---
    due = asm_ne[asm_ne.date <= t].groupby(["code_module","code_presentation"]).agg(n_due=("id_assessment","size"))
    coh = coh.merge(due.reset_index(), on=["code_module","code_presentation"], how="left"); coh["n_due"] = coh.n_due.fillna(0)
    s = a[a.date_submitted <= t]
    sg = s.groupby(KEY).agg(n_submitted=("id_assessment","size"), mean_score=("score","mean"),
                            mean_days_late=("late","mean"))
    s["wscore"] = s.score * s.weight / 100
    sw = s.groupby(KEY).wscore.sum(min_count=1).rename("weighted_score_sum")
    sdue = s[s.date <= t].groupby(KEY).size().rename("n_submitted_due")
    coh = coh.merge(sg.join([sw, sdue]).reset_index(), on=KEY, how="left")
    coh["n_submitted"] = coh.n_submitted.fillna(0); coh["n_submitted_due"] = coh.n_submitted_due.fillna(0)
    coh["n_missed"] = (coh.n_due - coh.n_submitted_due).clip(lower=0)
    coh["weighted_score_sum"] = coh.weighted_score_sum.fillna(0)
    # drop columns that must never be features (post-cutoff / label-derived)
    coh = coh.drop(columns=["date_unregistration"]).assign(cutoff=t)
    # keep final_result only for error analysis (not a feature)
    coh.to_csv(f"data/features_t{t}.csv", index=False)
    summary[t] = {"n": int(len(coh)), "dropout_rate": float(coh.dropout.mean()),
                  "n_dropout": int(coh.dropout.sum()), "n_features_cols": int(coh.shape[1])}
    print(t, summary[t], flush=True)

# store timing of unregistration for error analysis
sr[KEY+["date_unregistration"]].to_csv("data/unreg_dates.csv", index=False)
json.dump(summary, open("out/cohort_summary.json","w"), indent=1)
