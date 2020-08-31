[toc]



创新工厂2020AI训练营的工程实践项目，要求完成商品图像KPI指标的预测。

注：项目数据作为主办方的宝贵的资源，无法共享。本仓库仅分享个人在项目中编写的代码框架与通用的数据结构，仅做笔记所用。



# 1.项目功能

# 2.分析思路

# 3.代码框架

## 3.1数据存储层次

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gia3f3n8rsj30i004kaa4.jpg" alt="屏幕快照 2020-08-13 下午2.46.27" style="zoom: 50%;" />

1）sample _data存放图像数据

```
ls -hl ./sample_data/

-rw-r--r-- 1 pp pp   23K 8月  13 14:40 fd660039955e37374ed9b6f306e62832
-rw-r--r-- 1 pp pp   47K 8月  13 14:40 fe6bf7c1a1902a32c3dd5e1ebe57a4af
-rw-r--r-- 1 pp pp   68K 8月  13 14:40 fe86f0b1de8fb19c9c8d73bef4a9a7d0
-rw-r--r-- 1 pp pp   23K 8月  13 14:40 feabcf5ddd1c4879ce8bc1dc41047bf6
-rw-r--r-- 1 pp pp   81K 8月  13 14:40 ff8c6d370d8e205bd502434a9f46ee33
......
```

2）train_8000.csv 训练数据id+kpi, id 为图像的名字，对应的图像存储在sample _data 中

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp6v7iqx8j30k805sdh1.jpg" alt="屏幕快照 2020-08-13 下午2.53.21" style="zoom: 50%;" />

3)  test_2000.csv 训练数据id, id 为图像的名字，对应的图像存储在sample _data 中

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp6vhp7n3j30f605wt9n.jpg" alt="屏幕快照 2020-08-13 下午2.53.40" style="zoom:50%;" />

## 3.2 全链接网络+原始像素特征

cyy_nn.py

## 3.3 卷积网络+图像读入

cyy_07_lenet2.py

1.AdDataset() 基于torch.utils.data.Dataset 编写的数据接口，必选参数：csv_file, root_dir。AdDataset() 为一个可迭代对向，可以用下标索引/for 循环迭代；可以将AdDataset() 对象放入torch.utils.data.DataLoader()中，方便的使用其mini-batch和shuffle接口.

2.AlexNet_extra()为对AlexNet网络的改造类，可以返回conv5 256\*3\*3 张特征图对256求均值再拉伸为一个9维的特征向量。训练模型是特征向量不起作用，训练好模型后，可以使用该模型提取图像特征。



## 3.4 数据特征挖掘

**3.4.1依据kpi 划分数据集**

cyy_divide_dataset.py

运行程序后 deecamp文件夹下新增三个文件夹与三个cvs文件：

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gi5s8vhil8j30980bcmxy.jpg" style="zoom:50%;" />

**3.4.2 RGB特征+cnn特征**

cyy_extract_fea.py提取了每张图像的rgb均值与方差，以及图像9卷积特征。

**3.4.3 图像边缘信息**

cyy_extract_edg.py 用索贝尔算子提取图像边缘信息。

1.将彩色图像->灰度

2.用sobel算子权重初始化一个(1,1,3,3)卷积核

3.在图像上进行卷积操作，提取特征。

图像边缘信息提取效果图：

<img src="/Users/chenyingying01/Desktop/屏幕快照 2020-08-28 下午2.09.18.png" style="zoom: 33%;" />

**3.4.5特征简单分析**

cyy_extract_fea_analysis.py对提取的特征做一个简单的数值比较分析

| **KPI**  | (1,+∞)- good | **[0.3,1]**-normal | [0,0.3)-bad |
| -------- | ------------ | ------------------ | ----------- |
| 数据数量 | 610          | 2084               | 5306        |
| RGB均值  | 0.214        | 0.231              | 0.239       |
| RGB方差  | 0.585        | 0.595              | 0.597       |
| 边缘占比 | 0.708        | 0.787              | 0.802       |



## 3.5 fnn+12维特征

cyy_nn_post.py 使用提取的12维的特征，训练一个3层神经网络，用于拟合Kpi值。

## 3.6 模型融合

cyy_mode_mixtue.py

3.5 中训练的3层神经网络模型为一个基本模型，XGboost, Light_BGM 两个模型(队友训练)为另外两个基本模型。将三个模型的结果融合，作为最后的预测结果。

模型融合以rmse反比例作为融合置信系数。

