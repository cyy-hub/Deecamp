# 对提取的特征做一个简单的数值比较分析
import pandas as pd

train_feature = pd.read_csv("./deecamp_ad/train_feature.csv")
test_feature = pd.read_csv("./deecamp_ad/test_feature.csv")
train_data_good = train_feature[train_feature["kpi"] > 1]
train_data_bad = train_feature[train_feature["kpi"] < 0.3]
train_data_normal = train_feature[train_feature["kpi"] >= 0.3]
train_data_normal = train_data_normal[train_data_normal["kpi"] <= 1]

print(train_data_good["kpi"].shape, train_data_normal["kpi"].shape, train_data_bad["kpi"].shape)
print("rgb mean %.3f,%.3f,%.3f" % (
train_data_good["rgb_mean"].mean(), train_data_normal["rgb_mean"].mean(), train_data_bad["rgb_mean"].mean()))
print("rgb std %.3f,%.3f,%.3f" % (
train_data_good["rgb_std"].mean(), train_data_normal["rgb_std"].mean(), train_data_bad["rgb_std"].mean()))
print("edgproportion %.3f,%.3f,%.3f" % (
train_data_good["edg_proportion"].mean(), train_data_normal["edg_proportion"].mean(),
train_data_bad["edg_proportion"].mean()))
