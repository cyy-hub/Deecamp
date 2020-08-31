# cyy 20200726 提取图像的边缘占比
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
from PIL import Image, ImageSequence
import datetime
import matplotlib.pyplot as plt
import warnings
from cyy_cnn import AlexNet_extra
from cyy_cnn import AdDataset, Rescale, ToTensor, Normalize
import pdb

warnings.filterwarnings("ignore")
plt.ion()  # interactive mode
date = datetime.date.today()
data_str = date.strftime("%Y-%m-%d")


def edge_proportion(img):
    # 输入为一通道的灰度图片，再做边缘提取处理

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img.astype('uint8'),cmap='gray')
    im1 = torch.from_numpy(img.reshape((1, 1, img.shape[0], img.shape[1]))).float()
    conv1 = nn.Conv2d(1, 1, 3, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv1.weight.data = torch.from_numpy(sobel_kernel)
    edge1 = conv1(im1)
    edge1 = edge1.data.squeeze().numpy()
    all_pix = edge1.shape[0] * edge1.shape[1]
    non_zero_pix = np.count_nonzero(edge1)
    # plt.subplot(1,2,2)
    # plt.imshow(edge1,	cmap='gray')
    # plt.savefig("edge-2020-08-28.png")
    # pdb.set_trace()
    return non_zero_pix / all_pix


if __name__ == "__main__":
    train_set = AdDataset(csv_file="./deecamp_ad/train_8000.csv", root_dir="./deecamp_ad/sample_data/", Train=True,
                          Drop=False, Factor=1)
    test_set = AdDataset(csv_file="./deecamp_ad/test_2000_new.csv", root_dir="./deecamp_ad/sample_data/", Train=True,
                         Drop=False, Factor=1)
    # 读取已经提取的特征文件，在该文件中追加特征
    train_feature = pd.read_csv("./deecamp_ad/train_feature.csv")
    test_feature = pd.read_csv("./deecamp_ad/test_feature.csv")
    train_edge_pro = []
    for img, _ in train_set:
        img = img.mean(2)
        train_edge_pro.append(edge_proportion(img))
    train_feature["edg_proportion"] = np.array(train_edge_pro)
    test_edge_pro = []
    for img, _ in test_set:
        img = img.mean(2)
        test_edge_pro.append(edge_proportion(img))
    test_feature["edg_proportion"] = np.array(test_edge_pro)

    train_feature.to_csv("./deecamp_ad/train_feature.csv", index=False)
    test_feature.to_csv("./deecamp_ad/test_feature.csv", index=False)
