# cyy 20200718 基本卷积操作
from __future__ import print_function, division
import os
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
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
date=datetime.date.today()
data_str=date.strftime("%Y-%m-%d")


class AdDataset(Dataset):
    """Ad dataset."""

    def __init__(self, csv_file, root_dir, transform=None,Train=False,Drop=False,Factor=0.8):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Train=True设置训练集,调比例来设置是否全为训练集
        labels_kpi=pd.read_csv(csv_file)            # cvs kpi 文件
        train_num=int(len(labels_kpi)*Factor)
        if Train:
            if not Drop:
                self.labels_kpi =labels_kpi.iloc[:train_num,:].reset_index(drop=True)     
            else:
                labels_kpi=labels_kpi.iloc[:train_num,:]
                self.labels_kpi=labels_kpi[labels_kpi["kpi"]<5].reset_index(drop=True)
        else:
            self.labels_kpi = labels_kpi.iloc[train_num:,:].reset_index(drop=True)     

        self.root_dir = root_dir                    # 图片文件夹，里面放着所有图片
        self.transform = transform                  # 定义转换方法，

    def __len__(self):
        return len(self.labels_kpi)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.labels_kpi.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image=np.array(image)
        label_kpi= self.labels_kpi.iloc[idx, 1]
        label_kpi = np.array([label_kpi])
        label_kpi = label_kpi.astype('float')    
        sample = (image,label_kpi)                

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label_kpi = sample[0], sample[1]                         

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))               # 转换成0-1之间

        return (img,label_kpi)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label_kpi = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image).float(),
                torch.from_numpy(label_kpi).float())

class Normalize(object):
    def __call__(self,sample):
        image,label_kpi=sample[0],sample[1]
        image=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        return (image,label_kpi)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)            # 特征网络
        self.fc1 = nn.Linear(16 * 5 * 5, 120)       # 多层感知器聚合特征。
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))                 
        return x

class AlexNet(nn.Module):
    def __init__(self,num_class=2):
        nn.Module.__init__(self)
        
        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(3,96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),#对原变量进行覆盖
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
 
            #conv2
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
 
            #conv3
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
 
            #conv4
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
 
            #conv5
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)   
        )
        self.classifier = nn.Sequential(
            #fc6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
 
            #fc7
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
 
            #fc8
            nn.Linear(4096,num_class)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),256 * 6 * 6)
        x = self.classifier(x)
        return x 

class AlexNet_extra(nn.Module):
    def __init__(self,num_class=1):
        nn.Module.__init__(self)
        
        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(3,96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),#对原变量进行覆盖
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
 
            #conv2
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
 
            #conv3
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
 
            #conv4
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
 
            #conv5
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7,stride=3)   
        )
        self.classifier = nn.Sequential(
            #fc6
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
 
            #fc7
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
 
            #fc8
            nn.Linear(4096,num_class)
        )
    def forward(self,x):
        x = self.features(x)
        fea=x.mean(1)               # batchsize*256*3*3 对256张特征图求均值
        fea=fea.view(fea.size(0),-1).detach().tolist()
        x = x.view(x.size(0),256 * 3 * 3)
        x = self.classifier(x)
        return x,fea         

if __name__=="__main__":
    # 0.数据导入
    composed=transforms.Compose([Rescale((227,227)),ToTensor(),Normalize()])
    train_set=AdDataset(csv_file="./deecamp_ad/train_8000.csv",root_dir="./deecamp_ad/sample_data/",transform=composed,Train=True,Drop=True)
    test_set=AdDataset(csv_file="./deecamp_ad/train_8000.csv",root_dir="./deecamp_ad/sample_data/",transform=composed,Train=False,Drop=False)
    trainloader=DataLoader(train_set,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=128,shuffle=False)
    # 1.训练模型
    criterion=nn.MSELoss()
    net = AlexNet_extra()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # 
    l=0.0001
    optimizer = optim.Adam(net.parameters(),lr=l)
    max_loop=30
    test_losses,train_losses = [],[]
    delta=1
    for epoch in range(max_loop):  # loop over the dataset multiple times
        running_loss = 0.0
        test_loss=0.0
        net.train()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            #forward + backward + optimize
            outputs,fea = net(inputs)
            # print(outputs.size(),labels.size(),((outputs-labels)**2).size())
            # print(outputs.size(),len(fea[0]))
            # loss = criterion(outputs, labels)
            if abs(outputs-labels).float().mean()<=delta:                                   # 平方误差
                loss=((outputs-labels)**2).float().mean()
                # print("-"*25,epoch,i,abs(outputs-labels).float().mean().item(),loss.item())
            else:
                print("-"*50,epoch,i,abs(outputs-labels).float().mean().item(),loss.item()) # 绝对值误差
                loss=delta*abs(outputs-labels).float().mean()-0.5*delta**2
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    
                print('[train %d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                train_losses.append(running_loss / 5)
                running_loss = 0.0
        net.eval()
        if epoch%1==0:
            for i,data in enumerate(test_loader):
                inputs,labels=data
                outputs,fea=net(inputs)
                test_loss+=((outputs-labels)**2).sum().item()
            print("-"*5,"[test %d, %5d] loss: %.3f"%(epoch + 1, i+1, test_loss / len(test_set)))
            test_losses.append(test_loss / len(test_set))
            test_loss=0.0

    print("-"*50,'Finished Training',"-"*50)

    # 3.存模型
    model_save_path="./model_cnn_%s.pkl"%data_str
    torch.save(net,model_save_path)
    print("-"*50,'Model Saving',"-"*50)

    # 4.绘制训练指标图像
    print("plot figures")
    figure=plt.figure()
    axes1=figure.add_subplot(2,1,1)
    axes2 = figure.add_subplot(2, 1, 2)
    testloss_x=np.arange(0,len(test_losses))
    trainloss_x=np.arange(0,len(train_losses))*5
    axes1.plot(testloss_x[1:],test_losses[1:],label="test")
    #print(trainloss_x,train_losses)
    axes2.plot(trainloss_x[1:], train_losses[1:],label="train")
    axes1.set_xlabel("epoch")
    axes1.set_ylabel("test mse")
    axes2.set_xlabel("update times")
    axes2.set_ylabel("train mse")
    axes1.legend()
    axes2.legend()
    plt.savefig("test_%s_lr%s.png"%(data_str,l))



    #5.生成提交文件
    test_res=[]
    test_2000=AdDataset(csv_file="./deecamp_ad/test_2000_new.csv",root_dir="./deecamp_ad/sample_data/",transform=composed,Train=True,Drop=False,Factor=1)
    test_2000_loader=DataLoader(test_2000,batch_size=64,shuffle=False)
    net.eval()
    for i, data in enumerate(test_2000_loader):
        inputs,_=data
        outputs,fea=net(inputs)
        test_res+=outputs.detach().tolist()
    sub = pd.DataFrame()
    test_list=pd.read_csv("./deecamp_ad/test_2000_new.csv")
    sub['id'] = test_list['id']
    sub['kpi'] = np.array(test_res)
    sub.to_csv('submission_%s.csv'%data_str,index=False)
