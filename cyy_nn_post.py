import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image, ImageSequence
import numpy as np
import torch
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
import math
import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # todo 最后一层的激活函数,参数初始化
        return x


def data_load(path_csv, path_pic):
    """
    读入训练数据:train_data.csv
    :param path:train_data.csv 的路径
    :return: 训练数据：x(tensor),y(tensor)
    """
    train_data = pd.read_csv(path_csv)
    ##将图片转换成100*100的一维向量

    data_mean = train_data.groupby('id', as_index=False).mean()  # 相同ID分组求均值
    data_mean.drop(columns=['id'], inplace=True)  # 将ID那一列删除掉           # 这个地方应该很好实现随机打乱
    x_df = train_data[
        ["rgb_mean", "Anet fea0", "Anet fea1", "Anet fea2", "Anet fea3", "Anet fea4", "Anet fea5", "Anet fea6",
         "Anet fea7", "Anet fea8", "rgb_std", "edg_proportion"]]
    y_df = train_data['kpi']
    x_array = np.array(x_df)
    y_array = np.array(y_df)
    x = torch.tensor(x_array).float()
    y = torch.tensor(y_array).float().unsqueeze(1)
    # 直接读取，没有做数据归一化
    return x, y


if __name__ == "__main__":
    date = datetime.datetime.today()
    data_str = date.strftime("%Y-%m-%d")
    path_csv, path_pic = "./deecamp_ad/train_feature.csv", "./deecamp_ad/pictures/"
    x, y = data_load(path_csv, path_pic)
    num_test = len(x) // 5
    num_trian = len(x) - num_test
    # 第一次实验
    experiment_i = 4  # 0,1,2,3,4
    # pdb.set_trace()
    x_test = x[experiment_i * num_test:(experiment_i + 1) * num_test]
    x_train = torch.cat((x[0:experiment_i * num_test], x[(experiment_i + 1) * num_test:]), 0)
    y_test = y[experiment_i * num_test:(experiment_i + 1) * num_test]
    y_train = torch.cat((y[0:experiment_i * num_test], y[(experiment_i + 1) * num_test:]), 0)

    net = Net()
    # criterian=nn.MSELoss()
    # optimizer=optim.SGD(net.parameters(),lr=0.1,momentum=0.9)  # todo 带不带动量项,正则化技术，归一化处理
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    batch_size = 64
    maxloop = 50
    test_losses = []
    train_losses = []
    delta = 1
    for epoch in range(maxloop):
        randix = torch.randperm(num_trian)
        for i in range(num_trian // batch_size + 1):
            if i < num_trian // batch_size:
                inputs = x_train[randix[i * batch_size:(i + 1) * batch_size]]
                lables = y_train[randix[i * batch_size:(i + 1) * batch_size]]
            else:
                if i * batch_size < num_trian:
                    inputs = x_train[randix[i * batch_size:]]
                    lables = y_train[randix[i * batch_size:]]
            optimizer.zero_grad()
            outputs = net(inputs)
            if abs(outputs - lables).float().mean() <= delta:
                loss = ((outputs - lables) ** 2).float().mean()
            else:
                # print("-"*50,epoch,i,delta,abs(outputs-lables).float().mean().item())
                loss = delta * abs(outputs - lables).float().mean() - 0.5 * delta ** 2
            # loss=criterian(outputs,lables)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                rmse = math.sqrt(loss)
                print('[train %d, %5d] rmse: %.3f' %
                      (epoch + 1, i + 1, rmse))
                train_losses.append(rmse)
                train_loss = 0.0

        if epoch % 1 == 0:
            test_mini_batch = 160
            test_loss = 0.0
            for i in range(len(y_test) // test_mini_batch + 1):
                if i < len(y_test) // test_mini_batch:
                    test_inputs = x_test[i * test_mini_batch:(i + 1) * test_mini_batch]
                    test_labels = y_test[i * test_mini_batch:(i + 1) * test_mini_batch]
                else:
                    if i * test_mini_batch < len(y_test):
                        test_inputs = x_test[i * test_mini_batch:]
                        test_labels = y_test[i * test_mini_batch:]
                test_outputs = net(test_inputs)
                test_loss += ((test_outputs - test_labels) ** 2).sum().item()
            rmse = math.sqrt(test_loss / len(x_test))
            print("-" * 5, "[test %d, %5d] rmse: %.3f" % (epoch + 1, i + 1, rmse))
            test_losses.append(rmse)
    print("-" * 50, 'Finished Training', "-" * 50)

    # 3.存模型
    model_save_path = "./model_mlp_%s_%s.pkl" % (data_str, experiment_i)
    torch.save(net, model_save_path)
    print("-" * 50, 'Model Saving', "-" * 50)

    # 4.绘制训练指标图像
    print("plot figures")
    figure = plt.figure()
    axes1 = figure.add_subplot(2, 1, 1)
    axes2 = figure.add_subplot(2, 1, 2)
    testloss_x = np.arange(0, len(test_losses))
    trainloss_x = np.arange(0, len(train_losses)) * 10
    axes1.plot(testloss_x[0:], test_losses[0:], label="test")
    # print(trainloss_x,train_losses)
    axes2.plot(trainloss_x[0:], train_losses[0:], label="train")
    axes1.set_xlabel("epoch")
    axes1.set_ylabel("test rmse")
    axes2.set_xlabel("update times")
    axes2.set_ylabel("train rmse")
    axes1.legend()
    axes2.legend()
    print(experiment_i * num_test, (experiment_i + 1) * num_test)
    plt.savefig("test_%s_%s .png" % (data_str, experiment_i))

    # 5.生成提交文件
    test_res = []
    test_pd = pd.read_csv("./deecamp_ad/test_feature.csv")
    test_data = test_pd[
        ["rgb_mean", "Anet fea0", "Anet fea1", "Anet fea2", "Anet fea3", "Anet fea4", "Anet fea5", "Anet fea6",
         "Anet fea7", "Anet fea8", "rgb_std", "edg_proportion"]]
    test_array = np.array(test_data)
    test_data = torch.tensor(test_array).float()
    net.eval()
    test_mini_batch = 100  # 刚刚好整除
    for i in range(len(test_data) // test_mini_batch):
        test_inputs = test_data[i * test_mini_batch:(i + 1) * test_mini_batch]
        test_outputs = net(test_inputs)
        test_res += test_outputs.detach().tolist()
    sub = pd.DataFrame()
    sub['id'] = test_pd['id']
    sub['kpi'] = np.array(test_res)
    sub.to_csv('submission_fcl_%s.csv' % data_str, index=False)
