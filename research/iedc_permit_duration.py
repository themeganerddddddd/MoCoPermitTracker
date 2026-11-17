"""Permit filing-to-issuance prediction, with temporal splits and pre-filing features."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error
R=Path("research/iedc_results");R.mkdir(parents=True,exist_ok=True)
raw=pd.DataFrame(json.loads(Path("data/permits.json").read_text())["permits"])
a=pd.to_datetime(raw.addeddate,errors="coerce",utc=True);b=pd.to_datetime(raw.issueddate,errors="coerce",utc=True)
raw["approval_days"]=(b-a).dt.total_seconds()/86400
raw["filing_year"]=a.dt.year
raw["filing_month"]=a.dt.month
raw["issued_at"]=b
raw["valuation"]=pd.to_numeric(raw.declaredvaluation_num,errors="coerce")
raw["area"]=pd.to_numeric(raw.buildingarea_num,errors="coerce")
raw["val_log"]=np.log1p(raw.valuation.clip(lower=0))
raw["area_log"]=np.log1p(raw.area.clip(lower=0))
d=raw[(raw.approval_days.between(0,1500))&raw.filing_year.between(2001,2022)].copy()
features=["val_log","area_log","filing_year","filing_month","usecode","worktype"]
cat=["usecode","worktype"];num=[z for z in features if z not in cat]
rows=[];pred=[]
for year in [2018,2019,2020,2021,2022]:
 cutoff=pd.Timestamp(f"{year}-01-01",tz="UTC")
 tr=d[(d.filing_year<year)&(d.issued_at<cutoff)].copy()
 te=d[d.filing_year==year].copy()
 if len(tr)<1000 or len(te)<50:continue
 Xtr=tr[features].copy();Xte=te[features].copy()
 for c in cat:
  Xtr[c]=Xtr[c].fillna("UNKNOWN").astype(str);Xte[c]=Xte[c].fillna("UNKNOWN").astype(str)
 y=np.log1p(tr.approval_days)
 transform=ColumnTransformer([("num",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),num),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])
 A=transform.fit_transform(Xtr);B=transform.transform(Xte)
 median_overall=float(tr.approval_days.median())
 med_by_type=tr.groupby("worktype").approval_days.median().to_dict()
 models={"overall_median":np.full(len(te),median_overall),
         "worktype_median":te.worktype.map(med_by_type).fillna(median_overall).to_numpy()}
 for label,est in [("ridge",Ridge(alpha=70)),("gbdt",HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=180,learning_rate=.045,min_samples_leaf=35,l2_regularization=30,random_state=42))]:
  est.fit(A,y)
  models[label]=np.expm1(est.predict(B)).clip(0,1500)
 for name,yh in models.items():
  rows.append({"filing_year":year,"method":name,"n":len(te),"MAE_days":mean_absolute_error(te.approval_days,yh),"median_AE_days":median_absolute_error(te.approval_days,yh),"mean_actual_days":te.approval_days.mean()})
  for z,(idx,row) in enumerate(te.iterrows()):
   pred.append({"filing_year":year,"method":name,"worktype":row.worktype,"usecode":row.usecode,"actual_days":row.approval_days,"predicted_days":float(yh[z])})
pd.DataFrame(rows).to_csv(R/"permit_duration_backtest_by_year.csv",index=False)
p=pd.DataFrame(pred);p.to_csv(R/"permit_duration_predictions.csv",index=False)
if len(p):
 summary=p.groupby("method").apply(lambda a:pd.Series({"n":len(a),"MAE_days":np.mean(abs(a.actual_days-a.predicted_days)),"median_AE_days":np.median(abs(a.actual_days-a.predicted_days))}),include_groups=False).reset_index()
 summary.to_csv(R/"permit_duration_summary.csv",index=False)
 print(summary.to_string(index=False))
