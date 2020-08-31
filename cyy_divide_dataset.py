# 将图像数据集按照kpi大小划分为good,normal,bad 三个子集，以备后用
import os
import pandas as pd
import shutil

path_csv = "./deecamp_ad/train_8000.csv"
good_img_path = "./deecamp_ad/good"
bad_img_path = "./deecamp_ad/bad"
normal_img_path = "./deecamp_ad/normal"
if not os.path.isdir(good_img_path):
    os.mkdir(good_img_path)
if not os.path.isdir(bad_img_path):
    os.mkdir(bad_img_path)
if not os.path.isdir(normal_img_path):
    os.mkdir(normal_img_path)

train_data = pd.read_csv(path_csv)
train_data_good = train_data[train_data["kpi"] > 1]
train_data_bad = train_data[train_data["kpi"] < 0.3]
train_data_normal = train_data[train_data["kpi"] >= 0.3]
train_data_normal = train_data_normal[train_data_normal["kpi"] <= 1]

train_data_good.to_csv("./deecamp_ad/train_good.csv", index=False)
train_data_bad.to_csv("./deecamp_ad/train_bad.csv", index=False)
train_data_normal.to_csv("./deecamp_ad/train_normal.csv", index=False)

for img_name in train_data_good["id"]:
    source = "./deecamp_ad/sample_data/" + img_name
    shutil.copy(source, good_img_path)
for img_name in train_data_bad["id"]:
    source = "./deecamp_ad/sample_data/" + img_name
    shutil.copy(source, bad_img_path)
for img_name in train_data_normal["id"]:
    source = "./deecamp_ad/sample_data/" + img_name
    shutil.copy(source, normal_img_path)
