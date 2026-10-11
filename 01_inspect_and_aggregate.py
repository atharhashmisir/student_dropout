# 01_inspect_and_aggregate.py  -- run order: 1
# Data inspection + chunked aggregation of studentVle (no rows dropped).
import pandas as pd, numpy as np, json, sys
U = "/mnt/user-data/uploads/"
OUT = "out/"
rep = {"pandas": pd.__version__, "numpy": np.__version__}

names = ["assessments","courses","studentAssessment","studentInfo","studentRegistration","vle"]
dfs = {n: pd.read_csv(U+n+".csv") for n in names}
for n, d in dfs.items():
    rep[n] = {"rows": int(len(d)), "cols": list(d.columns),
              "missing": {c: int(v) for c, v in d.isna().sum().items() if v > 0},
              "full_duplicate_rows": int(d.duplicated().sum())}
si, sr, sa, asm, vle = (dfs[k] for k in ["studentInfo","studentRegistration","studentAssessment","assessments","vle"])
rep["studentInfo_key_dups"] = int(si.duplicated(["code_module","code_presentation","id_student"]).sum())
rep["studentReg_key_dups"] = int(sr.duplicated(["code_module","code_presentation","id_student"]).sum())
rep["vle_id_site_dups"] = int(vle.duplicated("id_site").sum())
rep["unique_students"] = int(si.id_student.nunique())
rep["final_result_counts"] = si.final_result.value_counts().to_dict()
rep["presentation_counts"] = si.groupby(["code_module","code_presentation"]).size().rename(lambda x: x).reset_index(name="n").to_dict("records")
m = si.merge(sr, on=["code_module","code_presentation","id_student"], how="left")
rep["crosstab_result_vs_unreg_present"] = pd.crosstab(m.final_result, m.date_unregistration.notna()).to_dict()
rep["withdrawn_unreg_date_summary"] = m.loc[m.final_result=="Withdrawn","date_unregistration"].describe().to_dict()
rep["sa_score_missing"] = int(sa.score.isna().sum())
rep["sa_banked"] = int(sa.is_banked.sum())
rep["assessments_type_counts"] = asm.assessment_type.value_counts().to_dict()
rep["assessments_date_missing_by_type"] = asm[asm.date.isna()].assessment_type.value_counts().to_dict()
rep["studied_credits_summary"] = si.studied_credits.describe().to_dict()
rep["imd_missing"] = int(si.imd_band.isna().sum())
for c in ["gender","age_band","disability","highest_education","imd_band","region"]:
    rep["dist_"+c] = si[c].value_counts(dropna=False).to_dict()

# --- studentVle: chunked aggregation (sum only; no row dropped) ---
site_map = vle.set_index("id_site")["activity_type"]
keys = ["code_module","code_presentation","id_student","date","activity_type"]
parts, tot_rows, tot_clicks, miss, unmatched, dup_rows = [], 0, 0, 0, 0, 0
dtypes = {"id_student":"int32","id_site":"int32","date":"int16","sum_click":"int32"}
for ch in pd.read_csv(U+"studentVle.csv", chunksize=1_000_000, dtype=dtypes):
    tot_rows += len(ch); tot_clicks += int(ch.sum_click.sum()); miss += int(ch.isna().sum().sum())
    ch["activity_type"] = ch.id_site.map(site_map)
    unmatched += int(ch.activity_type.isna().sum())
    ch["activity_type"] = ch.activity_type.fillna("unknown")
    g = ch.groupby(keys, as_index=False, observed=True).agg(clicks=("sum_click","sum"), n_rec=("sum_click","size"))
    parts.append(g)
daily = pd.concat(parts).groupby(keys, as_index=False).agg(clicks=("clicks","sum"), n_rec=("n_rec","sum"))
assert int(daily.clicks.sum()) == tot_clicks and int(daily.n_rec.sum()) == tot_rows
daily.to_csv("data/vle_daily_by_type.csv", index=False)
rep["studentVle"] = {"rows": tot_rows, "total_clicks": tot_clicks, "missing_cells": miss,
                     "records_without_vle_match": unmatched, "daily_activity_rows": int(len(daily)),
                     "date_min": int(daily.date.min()), "date_max": int(daily.date.max()),
                     "activity_types": sorted(daily.activity_type.unique().tolist())}
json.dump(rep, open(OUT+"inspection.json","w"), indent=1, default=str)
print(json.dumps(rep, indent=1, default=str)[:9000])
