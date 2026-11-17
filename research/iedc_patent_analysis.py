"""Reproducible patent technology-field + local employment prediction extension.
Source: NCSES NSB-2024-1 public-use workbook SWBINV-1; QCEW research pilot.
Grant-year inventors' county measures are retrospective research estimates.
"""
import re,io,json,time
from pathlib import Path
from collections import defaultdict
import requests, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

ROOT=Path("research/iedc_results");ROOT.mkdir(exist_ok=True,parents=True)
COUNTIES={"24031":"us-md-031","24021":"us-md-021","24027":"us-md-027","24033":"us-md-033","51059":"us-va-059","51107":"us-va-107","37183":"us-nc-183","25017":"us-ma-017"}
SRC="https://ncses.nsf.gov/pubs/nsb20241/assets/data/swbinv-1.xlsx"
report={"source":SRC}
def pull():
 r=requests.get(SRC,timeout=180);r.raise_for_status()
 xls=pd.ExcelFile(io.BytesIO(r.content))
 rows=[];tech_names={}
 for sh in xls.sheet_names:
  if sh in ("Index","Table SWBINV1-1","Table SWBINV1-38"):continue
  f=pd.read_excel(xls,sheet_name=sh,header=3)
  # Headers row 4 in source, data row 5 onward.
  if len(f.columns)<20:continue
  first=f.columns[0];f[first]=f[first].astype(str).str.lower()
  f=f[f[first].isin(set(COUNTIES.values()))]
  title=pd.read_excel(xls,sheet_name=sh,header=None,nrows=2).iloc[1,0]
  tech=re.sub("USPTO utility patents granted to U.S. inventors (?:in )?","",str(title)).split(", by U.S. county")[0].strip()
  tech_names[sh]=tech
  for _,r in f.iterrows():
   code=next((k for k,v in COUNTIES.items() if v==str(r[first])),None)
   if not code:continue
   for c in f.columns[1:]:
    try:y=int(float(c));val=float(r[c])
    except:continue
    if 1998<=y<=2022 and np.isfinite(val):
     rows.append({"fips":code,"year":y,"sheet":sh,"technology":tech,"patents":val})
 df=pd.DataFrame(rows)
 df.to_csv(ROOT/"patent_county_technology_1998_2022.csv",index=False)
 report["extracted"]={"rows":len(df),"n_counties":df.fips.nunique(),"n_years":df.year.nunique(),"technologies":tech_names}
 return df
def backtest(p):
 if not (ROOT/"county_industry_qcew.csv").exists():return
 q=pd.read_csv(ROOT/"county_industry_qcew.csv",dtype={"fips":str,"industry":str})
 q=q.sort_values(["fips","industry","year"]).reset_index(drop=True)
 g=q.groupby(["fips","industry"],sort=False)
 q["growth1"]=100*np.log(q.employment/g.employment.shift(1))
 q["growth2past"]=100*np.log(q.employment/g.employment.shift(2))
 q["future"]=100*np.log(g.employment.shift(-2)/q.employment)
 q["netjobs"]=g.employment.shift(-2)-q.employment
 q["size"]=np.log1p(q.employment)
 q["wage"]=np.log1p(q.wages)
 q["est"]=np.log1p(q.establishments)
 q["share"]=q.employment/q.groupby(["fips","year"]).employment.transform("sum")
 q["emerging"]=((q["future"]>=100*np.log(1.15))&(q.netjobs>=100)&(q.employment>=250)).astype(int)
 # One column per WIPO technology. Do not assert technology == company NAICS.
 pp=p.pivot_table(index=["fips","year"],columns="technology",values="patents",aggfunc="sum").reset_index()
 alltech=[c for c in pp.columns if c not in ("fips","year") and "by U.S." not in c]
 pp[alltech]=np.log1p(pp[alltech].clip(lower=0))
 pat_total=[c for c in alltech if "1998" in c or c.lower().startswith("by u.s.")]
 # Total patent sheet gets no special treatment if category name is ambiguous.
 pp["all_patents_log"]=pp[alltech].sum(axis=1)/max(1,len(alltech))
 d=q.merge(pp,on=["fips","year"],how="left")
 ordinary=["growth1","growth2past","size","wage","est","share"]
 patent=["all_patents_log"]+[c for c in alltech if c!= "all_patents_log"]
 results=[];pred=[]
 for yr in [2020,2021,2022]:
  # Eligibility: patent observations from grants already published by the end of origin yr.
  # Data require subsequent outcome 2 years later. Label training cutoff prevents target look-ahead.
  tr=d[(d.year<=yr-3)&(d.year>=2016)&d.future.notna()&d.growth2past.notna()&(d.employment>=250)]
  te=d[(d.year==yr)&d.future.notna()&d.growth2past.notna()&(d.employment>=250)]
  if len(tr)<100 or len(te)<20:continue
  models=[
   ("ridge_econ",ordinary,"ridge"),
   ("gbdt_econ",ordinary,"gbdt"),
   ("ridge_patents",ordinary+patent,"ridge"),
   ("gbdt_patents",ordinary+patent,"gbdt")]
  for name,cols,typ in models:
   X=tr[cols].replace([np.inf,-np.inf],np.nan)
   T=te[cols].replace([np.inf,-np.inf],np.nan)
   im=SimpleImputer(strategy="median",add_indicator=False,keep_empty_features=True)
   X=im.fit_transform(X);T=im.transform(T)
   if typ=="ridge":mod=make_pipeline(StandardScaler(),Ridge(alpha=20))
   else:mod=HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.04,l2_regularization=40,random_state=42)
   mod.fit(X,tr.future)
   yh=mod.predict(T)
   for i in range(len(te)):
    row=te.iloc[i]
    pred.append({"year":yr,"target_year":yr+2,"fips":row.fips,"sector":row.sector,"naics":row.industry,"employment":row.employment,"method":name,"actual_growth_2y":float(row.future),"predicted_growth_2y":float(yh[i]),"emerging":int(row.emerging)})
   results.append({"origin_year":yr,"method":name,"n":len(te),"MAE_growth_pp":mean_absolute_error(te.future,yh),"RMSE_growth_pp":float(np.mean((te.future-yh)**2)**.5)})
 pd.DataFrame(results).to_csv(ROOT/"patent_industry_backtest_by_year.csv",index=False)
 out=pd.DataFrame(pred);out.to_csv(ROOT/"patent_industry_predictions.csv",index=False)
 if len(out):
  summ=[]
  for method,frame in out.groupby("method"):
   err=np.abs(frame.actual_growth_2y-frame.predicted_growth_2y)
   alltop=[]
   for yr,grp in frame.groupby("year"):
    top=grp.nlargest(min(25,len(grp)),"predicted_growth_2y")
    alltop.extend(top.emerging.tolist())
   summ.append({"method":method,"n":len(frame),"mae_pp":float(err.mean()),"rmse_pp":float(np.sqrt(np.mean(err**2))),"actual_emergence_rate":float(frame.emerging.mean()),"precision_top25":float(np.mean(alltop)) if alltop else None})
  pd.DataFrame(summ).to_csv(ROOT/"patent_industry_backtest_summary.csv",index=False)
  report["model_comparison"]=summ
  for yr in [2022]:
   ex=out[(out.year==yr)&(out.method=="gbdt_patents")].nlargest(20,"predicted_growth_2y")
   ex.to_csv(ROOT/"patent_top_predicted_industries_2022.csv",index=False)
def main():
 try:
  p=pull();backtest(p)
 except Exception as e:
  report["error"]=str(e)
  import traceback;report["traceback"]=traceback.format_exc()
 (ROOT/"patent_analysis_status.json").write_text(json.dumps(report,indent=2,default=str))
 print(json.dumps({k:v for k,v in report.items() if k!="extracted"},indent=2,default=str))
if __name__=="__main__":main()
