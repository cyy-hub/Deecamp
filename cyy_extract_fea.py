# cyy 20200722 提取转存数据可量化特征
from __future__ import print_function, division
import os
import shutil
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageSequence
import datetime
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
from cyy_cnn import AlexNet_extra
from cyy_cnn import AdDataset,  Rescale, ToTensor, Normalize

warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
date=datetime.date.today()
data_str=date.strftime("%Y-%m-%d")

if __name__=="__main__":

    composed=transforms.Compose([Rescale((227,227)),ToTensor(),Normalize()])
    train_set=AdDataset(csv_file="./deecamp_ad/train_8000.csv",root_dir="./deecamp_ad/sample_data/",transform=composed,Train=True,Drop=False,Factor=1)
    trainloader=DataLoader(train_set,batch_size=128,shuffle=False)
    # 原始test_2000.csv 无KPI信息，为了方便数据接口编写，填充一列为0的KPI数据，命名为test_2000_new.csv
    test_set=AdDataset(csv_file="./deecamp_ad/test_2000_new.csv",root_dir="./deecamp_ad/sample_data/",transform=composed,Train=True,Drop=False,Factor=1)
    testloader=DataLoader(test_set,batch_size=128,shuffle=False)
    train_csv=pd.read_csv("./deecamp_ad/train_8000.csv")
    test_csv=pd.read_csv("./deecamp_ad/test_2000.csv")

    train_feature=pd.DataFrame()
    test_feature = pd.DataFrame()
    train_feature["id"]=train_csv["id"]
    test_feature["id"]=test_csv["id"]

    # step1: extract rgb mean
    rgb_mean, rgb_mean_test = [], []
    rgb_std, rgb_std_test = [], []
    for img,kpi in train_set:
        rgb_mean.append(img.mean())
    train_feature["rgb_mean"]=np.array(rgb_mean)
    for img,kpi in test_set:
        rgb_mean_test.append(img.mean())
    test_feature["rgb_mean"]=np.array(rgb_mean_test)
    
    # step2: extract rgb std
    for img,kpi in train_set:
        rgb_std.append(img.std())
    train_feature["rgb_std"]=np.array(rgb_std)
    for img,kpi in test_set:
        rgb_std_test.append(img.std())
    test_feature["rgb_std"]=np.array(rgb_std_test)


    # step3: extract cnn feature
    # 训练集特征
    PATH="./model_cnn_2020-07-23.pkl"     # 用于提取特征的网络
    net=torch.load(PATH)
    net.eval()
    model_output,model_fea=[],[]
    for i, data in enumerate(trainloader):
        inputs,_=data
        outputs,fea=net(inputs)
        model_output+=outputs.detach().tolist()
        model_fea+=fea
    train_feature["Anet out"]=np.array(model_output)
    model_fea=np.array(model_fea)
    for i in range(len(model_fea[0])):
        feai=[]
        for j in range(len(model_fea)):
            feai.append(model_fea[j][i])
        train_feature["Anet fea%s"%i]=np.array(feai)
    # 测试集特征 
    model_output_test,model_fea_test=[],[]
    for i, data in enumerate(testloader):
        # print(i)
        inputs,_=data
        outputs,fea=net(inputs)
        model_output_test+=outputs.detach().tolist()
        model_fea_test+=fea
    test_feature["Anet out"]=np.array(model_output_test)
    model_fea_test=np.array(model_fea_test)
    for i in range(len(model_fea_test[0])):
        feai=[]
        for j in range(len(model_fea_test)):
            feai.append(model_fea_test[j][i])
        test_feature["Anet fea%s"%i]=np.array(feai)

    # step 4: write to cvs file
    train_feature.to_csv("./deecamp_ad/train_feature.csv",index=False)
    test_feature.to_csv("./deecamp_ad/test_feature.csv",index=False)
