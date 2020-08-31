# cyy 20200723 模型融合文件
import numpy as np
import pandas as pd
import datetime

date = datetime.date.today()
data_str = date.strftime("%Y-%m-%d")

res_fcl = pd.read_csv("./submission_fcl_new.csv")
res_xgboost = pd.read_csv("./submission_xgboost_new.csv")
res_lightGBM = pd.read_csv("./submission_lightGBM_new.csv")
sub = pd.DataFrame()
sub["id"] = res_fcl["id"]
# 权重由三个模型的5折交叉验证平均rmse反比确定
fac_fcl, fac_xgboost, fac_lightGBM = 0.288, 0.329, 0.383

sub["kpi"] = fac_fcl * res_fcl["kpi"] + fac_xgboost * res_xgboost["kpi"] + fac_lightGBM * res_lightGBM["kpi"]
sub.to_csv("submission_%s.csv" % data_str, index=False)
