"""IEDC reproducible public-data forecasting pilot. No private credentials."""
import os, io, csv, json, re, time, warnings, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT=Path("research/iedc_results"); ROOT.mkdir(parents=True,exist_ok=True)
AREAS={"24031":"Montgomery MD","24021":"Frederick MD","24027":"Howard MD","24033":"Prince Georges MD","51059":"Fairfax VA","51107":"Loudoun VA","37183":"Wake NC","25017":"Middlesex MA"}
YEARS=range(2009,2026)
IND={"11":"Agriculture","21":"Mining","22":"Utilities","23":"Construction","31-33":"Manufacturing","42":"Wholesale","44-45":"Retail","48-49":"Transport","51":"Information","52":"Finance","53":"RealEstate","54":"ProfessionalTech","55":"Management","56":"Admin","61":"Education","62":"Healthcare","71":"Arts","72":"Hospitality","81":"OtherServices"}
S=requests.Session();S.headers.update({"User-Agent":"iedc-public-research-replication/1.0"})
status={"started":pd.Timestamp.utcnow().isoformat(),"data_sources":{},"errors":[]}
def fetch_one(pair):
 fips,year=pair
 url=f"https://data.bls.gov/cew/data/api/{year}/a/area/{fips}.csv"
 err=""
 for k in range(3):
  try:
   r=requests.get(url,timeout=45,headers={"User-Agent":"Academic research data retrieval (contact: GitHub issue)"})
   r.raise_for_status()
   raw=pd.read_csv(io.StringIO(r.text),dtype={"area_fips":str,"industry_code":str,"own_code":str,"size_code":str},low_memory=False)
   raw=raw[(raw.area_fips==fips)&(raw.own_code=="5")&(raw.size_code=="0")&raw.industry_code.isin(IND)]
   if raw.empty: return [],"empty QCEW rows "+url
   rows=[]
   for _,x in raw.iterrows():
    if str(x.get("disclosure_code","")).strip() not in ("","nan"):continue
    try:emp=float(x.annual_avg_emplvl)
    except:continue
    if emp<=0:continue
    rows.append({"fips":fips,"county":AREAS[fips],"year":year,"industry":str(x.industry_code),"sector":IND[str(x.industry_code)],"employment":emp,"establishments":pd.to_numeric(x.get("annual_avg_estabs"),errors="coerce"),"wages":pd.to_numeric(x.get("avg_annual_pay"),errors="coerce")})
   return rows,None
  except Exception as e:
   err=str(e);time.sleep(k+1)
 return [],url+" => "+err
def build_qcew():
 tasks=[(f,y) for f in AREAS for y in YEARS];rows=[]
 with ThreadPoolExecutor(max_workers=10) as pool:
  for fut in as_completed([pool.submit(fetch_one,task) for task in tasks]):
   dat,err=fut.result();rows+=dat
   if err:status["errors"].append(err)
 df=pd.DataFrame(rows)
 status["data_sources"]["qcew"]={"rows":len(df),"years":sorted(df.year.unique().tolist()) if len(df) else [],"counties":len(df.fips.unique()) if len(df) else 0}
 if len(df):df.to_csv(ROOT/"county_industry_qcew.csv",index=False)
 return df
def permits():
 p=Path("data/permits.json")
 if not p.exists(): return pd.DataFrame()
 blob=json.loads(p.read_text());raw=pd.DataFrame(blob["permits"])
 raw["added"]=pd.to_datetime(raw["addeddate"],errors="coerce",utc=True)
 raw["issued"]=pd.to_datetime(raw["issueddate"],errors="coerce",utc=True)
 raw["finaled"]=pd.to_datetime(raw["finaleddate"],errors="coerce",utc=True)
 raw["approval_days"]=(raw.issued-raw.added).dt.total_seconds()/86400
 raw["total_days"]=(raw.finaled-raw.added).dt.total_seconds()/86400
 raw["issue_year"]=raw.issued.dt.year
 raw["filing_year"]=raw.added.dt.year
 raw["valuation"]=pd.to_numeric(raw.declaredvaluation_num,errors="coerce")
 filt=raw[raw.approval_days.between(0,6000)].copy()
 desc=filt.groupby("worktype").approval_days.agg(["size","median",lambda x:x.quantile(.9)]).reset_index()
 desc.columns=["worktype","n","median_days","p90_days"]
 desc.to_csv(ROOT/"permit_type_distribution.csv",index=False)
 life=filt.groupby("usecode").approval_days.agg(["size","median",lambda x:x.quantile(.9)]).reset_index()
 life.columns=["usecode","n","median_days","p90_days"]
 life.to_csv(ROOT/"permit_use_distribution.csv",index=False)
 annual=filt.groupby("issue_year").agg(issued=("approval_days","size"),median_days=("approval_days","median"),p90_days=("approval_days",lambda x:x.quantile(.9)),valuation=("valuation","sum")).reset_index().rename(columns={"issue_year":"year"})
 annual.to_csv(ROOT/"permits_annual.csv",index=False)
 status["data_sources"]["permits"]={"all_rows":len(raw),"issued_intervals":len(filt),"median_days":float(filt.approval_days.median())}
 return annual
def patent_source():
 """Locate NCSES public workbook via HTML links; no invented fallback data."""
 u="https://ncses.nsf.gov/pubs/nsb20241/data"
 try:
  r=requests.get(u,timeout=40);r.raise_for_status()
  html=r.text
  hits=re.findall(r"""(?:href|url)\s*[:=]\s*["']([^"']+\.xlsx(?:\?[^"']*)?)["']""",html,re.I)
  hits=[urljoin(u,x.replace("&amp;","&")) for x in hits if "swbinv" in x.lower() or "workbook" in x.lower()]
  status["data_sources"]["patent_workbook_discovery"]={"candidate_urls":hits[:10],"html_bytes":len(html)}
  # The workbook is only used if we can verify structure and county/field/time columns.
  for link in hits[:8]:
   try:
    res=requests.get(link,timeout=90);res.raise_for_status()
    if len(res.content)<20000:continue
    exc=pd.ExcelFile(io.BytesIO(res.content))
    status["data_sources"]["patent_workbook_discovery"]["sheets"]=exc.sheet_names[:20]
    previews={}
    for sh in exc.sheet_names:
     p=pd.read_excel(exc,sheet_name=sh,header=None,nrows=9)
     previews[sh]={"columns":p.shape[1],"sample":p.fillna("").astype(str).iloc[:9,:12].values.tolist()}
    (ROOT/"patent_workbook_sheets_preview.json").write_text(json.dumps(previews,indent=2))
    for sheet in exc.sheet_names:
     sample=pd.read_excel(exc,sheet_name=sheet,nrows=12,header=None)
     s=sample.astype(str).to_string(index=False,header=False).lower()
     if "county" in s and ("field" in s or "technology" in s):
      (ROOT/"patent_workbook_downloaded.xlsx").write_bytes(res.content)
      status["data_sources"]["patents"]={"path":"patent_workbook_downloaded.xlsx","source":link,"sheets":exc.sheet_names}
      return
   except Exception as e:status["errors"].append("patent candidate: "+str(e))
 except Exception as e:status["errors"].append("patent discovery: "+str(e))
 status["data_sources"]["patents"]={"status":"not_downloaded_no_usable_link_or_schema"}
def evaluate(q):
 if q.empty:return
 q=q.sort_values(["fips","industry","year"]).copy()
 g=q.groupby(["fips","industry"],sort=False)
 q["growth_1y"]=100*np.log(q["employment"]/g.employment.shift(1))
 q["growth_2y_forward"]=100*np.log(g.employment.shift(-2)/q.employment)
 q["growth_prev_2y"]=100*np.log(q.employment/g.employment.shift(2))
 q["emplog"]=np.log(q.employment)
 q["wage_log"]=np.log(q.wages.where(q.wages>0))
 q["estlog"]=np.log1p(q.establishments)
 q["lqproxy"]=q.employment / q.groupby(["fips","year"]).employment.transform("sum")
 q["emerging"]=((q.growth_2y_forward>=np.log(1.15)*100)&((g.employment.shift(-2)-q.employment)>=100)&(q.employment>=250))
 feats=["growth_1y","growth_prev_2y","emplog","wage_log","estlog","lqproxy"]
 d=q.dropna(subset=["growth_2y_forward","growth_1y","growth_prev_2y"]).copy()
 test=d[(d.year>=2019)&(d.year<=2023)].copy()
 results=[];predictions=[]
 for y in sorted(test.year.unique()):
  train=d[d.year<=y-3].copy()
  testy=test[test.year==y].copy()
  if len(train)<200 or len(testy)<15:continue
  X=train[feats].replace([np.inf,-np.inf],np.nan)
  T=testy[feats].replace([np.inf,-np.inf],np.nan)
  models={"persistence":None,
          "ridge":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=20)),
          "gbdt":HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=12,min_samples_leaf=25,l2_regularization=20,learning_rate=.04,random_state=42)}
  for name,m in models.items():
   if m is None:pr=testy.growth_1y.to_numpy()
   else:
    xx=X.copy().replace([np.inf,-np.inf],np.nan)
    tt=T.copy().replace([np.inf,-np.inf],np.nan)
    if name=="gbdt":
     xx=xx.fillna(X.median()).fillna(0)
     tt=tt.fillna(X.median()).fillna(0)
    m.fit(xx,train.growth_2y_forward)
    pr=m.predict(tt)
   for ix,v in enumerate(pr):
    obj=testy.iloc[ix]
    predictions.append({"origin":y,"target_year":y+2,"fips":obj.fips,"industry":obj.industry,"method":name,"actual_growth_2y":obj.growth_2y_forward,"predicted_growth_2y":v,"emerging":bool(obj.emerging)})
   results.append({"origin":y,"method":name,"n":len(pr),"mae_pp":mean_absolute_error(testy.growth_2y_forward,pr)})
 if not results:return
 r=pd.DataFrame(results);r.to_csv(ROOT/"industry_backtest_by_year.csv",index=False)
 pd.DataFrame(predictions).to_csv(ROOT/"industry_predictions.csv",index=False)
 summary=r.groupby("method").apply(lambda a:pd.Series({"test_predictions":a.n.sum(),"weighted_mae_pp":np.average(a.mae_pp,weights=a.n)}),include_groups=False).reset_index()
 summary.to_csv(ROOT/"industry_backtest_summary.csv",index=False)
 status["industry_results"]=summary.to_dict(orient="records")
def evaluate_permit_addition(q,annual):
 if q.empty or annual.empty:return
 # Single-county test; too few independent annual points for credible ML inference.
 subset=q[(q.fips=="24031")&(q.industry=="23")].sort_values("year")
 x=subset.merge(annual,on="year",how="left")
 x["growth_future"]=100*np.log(x.employment.shift(-1)/x.employment)
 x["growth_lag"]=100*np.log(x.employment/x.employment.shift(1))
 x=x.dropna(subset=["growth_future","growth_lag","median_days","issued"])
 preds=[]
 for y in range(2018,2025):
  train=x[x.year<=y-2];test=x[x.year==y]
  if len(train)<6 or len(test)!=1:continue
  base=["growth_lag"];plus=["growth_lag","issued","median_days", "valuation"]
  for name,cols in [("employment_only",base),("employment_plus_permits",plus)]:
   mod=make_pipeline(StandardScaler(),Ridge(alpha=10))
   mod.fit(train[cols].fillna(0),train.growth_future)
   for val in mod.predict(test[cols].fillna(0)):
    preds.append({"year":y,"outcome_year":y+1,"method":name,"actual_growth":float(test.growth_future.iloc[0]),"predicted_growth":float(val)})
 if preds:
  pr=pd.DataFrame(preds);pr.to_csv(ROOT/"construction_permit_backtest.csv",index=False)
  status["permit_forecast_results"]=pr.groupby("method").apply(lambda x:{"n":len(x),"mae_pp":float(np.mean(np.abs(x.actual_growth-x.predicted_growth)))},include_groups=False).to_dict()
def main():
 q=build_qcew();a=permits();patent_source()
 if len(q):evaluate(q);evaluate_permit_addition(q,a)
 status["finished"]=pd.Timestamp.utcnow().isoformat()
 (ROOT/"status.json").write_text(json.dumps(status,indent=2,default=str))
 print(json.dumps(status,indent=2,default=str))
if __name__=="__main__":
 try:main()
 except Exception as exc:
  status["error_fatal"]=str(exc);status["traceback"]=traceback.format_exc()
  (ROOT/"status.json").write_text(json.dumps(status,indent=2,default=str));raise
