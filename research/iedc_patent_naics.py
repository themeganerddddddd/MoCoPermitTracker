"""Exploratory WIPO technology-to-NAICS sector hypotheses, not an official concordance.
Tests incremental predictive value with industry-specific patent features.
"""
import json,numpy as np,pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
R=Path("research/iedc_results")
# A one-to-many expert hypothesis. Not a USPTO-validated mapping.
MAP={
"22":["electrical machinery","environmental technology","thermal processes"],
"23":["civil engineering","environmental technology","handling"],
"31-33":["semiconductors","materials, metallurgy","machine tools","chemical engineering","engines, pumps","macromolecular chemistry","electrical machinery","mechanical elements","transport","surface technology","food chemistry","biotechnology","pharmaceuticals"],
"51":["telecommunications","digital communication","computer technology","audio-visual","basic communication","IT methods"],
"52":["IT methods for management","computer technology"],
"54":["computer technology","IT methods","biotechnology","pharmaceuticals","medical technology","measurement","analysis of biological materials","civil engineering","semiconductors","environmental technology"],
"62":["medical technology","biotechnology","pharmaceuticals","analysis of biological materials"],
"56":["control","IT methods for management"],
"48-49":["transport","engines, pumps","mechanical elements"]
}
q=pd.read_csv(R/"county_industry_qcew.csv",dtype={"fips":str,"industry":str}).sort_values(["fips","industry","year"])
p=pd.read_csv(R/"patent_county_technology_1998_2022.csv",dtype={"fips":str})
p=p[~p.technology.str.startswith("USPTO",na=False)]
p["tech_group"]=p.technology.str.lower()
def matched(code,tech):
 return float(any(word in tech.lower() for word in MAP.get(code,[])))
p["patents"]=pd.to_numeric(p.patents,errors="coerce").fillna(0)
keys=q[["fips","industry","year"]].drop_duplicates()
pr=p.merge(keys,on=["fips","year"],how="inner")
pr["weighted_patents"]=pr.apply(lambda x:x.patents*matched(x.industry,x.tech_group),axis=1)
a=pr.groupby(["fips","industry","year"],as_index=False).agg(matched_patents=("weighted_patents","sum"),all_tech_patents=("patents","sum"))
a["patent_share"]=a.matched_patents/(a.all_tech_patents+1)
a["patent_log"]=np.log1p(a.matched_patents.clip(lower=0))
a.to_csv(R/"technology_naics_exploratory_crosswalk_features.csv",index=False)
g=q.groupby(["fips","industry"],sort=False)
q["past_1"]=100*np.log(q.employment/g.employment.shift(1))
q["past_2"]=100*np.log(q.employment/g.employment.shift(2))
q["target"]=100*np.log(g.employment.shift(-2)/q.employment)
q["future_jobs"]=g.employment.shift(-2)-q.employment
q["emerging"]=((q.target>=100*np.log(1.15))&(q.future_jobs>=100)&(q.employment>=250)).astype(int)
q["size"]=np.log1p(q.employment)
q["wage"]=np.log1p(q.wages)
q["est"]=np.log1p(q.establishments)
q["county_share"]=q.employment/q.groupby(["fips","year"]).employment.transform("sum")
q=q.merge(a,on=["fips","industry","year"],how="left")
isector=pd.get_dummies(q.industry,prefix="naics",dtype=int)
q=pd.concat([q.reset_index(drop=True),isector.reset_index(drop=True)],axis=1)
base=["past_1","past_2","size","wage","est","county_share"]+list(isector.columns)
plus=base+["patent_log","patent_share"]
pred=[]
for year in [2020,2021,2022]:
 train=q[(q.year>=2016)&(q.year<=year-3)&q.past_2.notna()&q.target.notna()&(q.employment>=250)]
 test=q[(q.year==year)&q.past_2.notna()&q.target.notna()&(q.employment>=250)]
 if len(train)<100 or len(test)<20:continue
 for name,features,kind in [("gbdt_with_naics",base,"gbdt"),("gbdt_with_matched_patents",plus,"gbdt"),("ridge_with_naics",base,"ridge"),("ridge_with_matched_patents",plus,"ridge")]:
  imp=SimpleImputer(strategy="median",keep_empty_features=True)
  X=imp.fit_transform(train[features]);T=imp.transform(test[features])
  if kind=="gbdt": model=HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.04,l2_regularization=40,random_state=42)
  else:model=make_pipeline(StandardScaler(),Ridge(alpha=20))
  model.fit(X,train.target);yhat=model.predict(T)
  for i,(_,x) in enumerate(test.iterrows()):
   pred.append({"origin":year,"method":name,"fips":x.fips,"naics":x.industry,"actual_growth":x.target,"predicted_growth":float(yhat[i]),"emerging":int(x.emerging)})
df=pd.DataFrame(pred);df.to_csv(R/"patent_naics_predictions.csv",index=False)
summ=[]
for name,d in df.groupby("method"):
 precision=[]
 for year,z in d.groupby("origin"):
  precision.extend(z.nlargest(25,"predicted_growth").emerging.tolist())
 summ.append({"model":name,"n":len(d),"mae_pp":float(np.mean(abs(d.actual_growth-d.predicted_growth))),"precision_top25":float(np.mean(precision))})
pd.DataFrame(summ).to_csv(R/"patent_naics_comparison.csv",index=False)
(R/"patent_naics_crosswalk_hypotheses.json").write_text(json.dumps({"caveat":"Illustrative WIPO technology-related-to-NAICS hypotheses; not the USPTO ALP patent-industry concordance.", "map":MAP},indent=2))
print(pd.DataFrame(summ).to_string(index=False))
