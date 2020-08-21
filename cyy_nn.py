import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image,ImageSequence
import numpy as np
import torch
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(30000,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,1)           

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        # kpi ->[0,+inf] 
        x=self.fc3(x)                       
        return  x

def feature_extract(data, is_train):
    """
    :type  data: pd-["id", "kpi"]
    :type is_train: bool
    :rtype:  [list]  list-[id, feature, kpi] 
    """
    data_feed = []
    error_list = []
    for file_name in data['id'].tolist():
        image_path = f'./deecamp_ad/sample_data/{file_name}'
        # image_path = f'./pictures/{file_name}'
        try:
            img = Image.open(image_path)
            frames = []
            # 训练数据中存在动图GIF，逐帧读取图像
            for frame in ImageSequence.Iterator(img):
                f = frame.copy()
                # 逐张图像先resize,再拉平成一维度向量
                frames.append(np.array(f.resize((100, 100)).convert('RGB')))
                features = frames[0].flatten().tolist()
                if is_train:
                    # 将对应的kpi 取出来放到特征的尾部
                    features.append(data.loc[data['id'].isin([file_name])]['kpi'].tolist()[0])
                # 将对应的id取出，放在feature List 的开端
                features.insert(0,file_name)
                data_feed.append(features)
        except:
            error_list.append(file_name)
            continue
    return data_feed

def data_loader(train_path, test_path):
    # read training and testing csv data
    train_pd = pd.read_csv(train_data_path)
    test_pd = pd.read_csv(test_data_path)

    train_fea= feature_extract(train_pd,is_train = True)
    train_fea= pd.DataFrame(train_fea)
    
    train_fea.rename(columns={0:'id',30001:'kpi'},inplace=True)
    train_fea_mean = train_fea.groupby('id',as_index = False).mean()
    train_fea_mean=train_fea_mean[train_fea_mean["kpi"]<5].reset_index(drop=True)
    # remove id column
    train_fea_mean.drop(columns=['id'],inplace=True)
    train_x_df= train_fea_mean.iloc[:,:-1]
    train_y_df = train_fea_mean['kpi']
    train_x_array = np.array(train_x_df)
    train_x = torch.tensor(train_x_array).float()
    train_x = train_x / 255.0                                                          
    train_y_array=np.array(train_y_df)
    train_y=torch.tensor(train_y_array).float().unsqueeze(1)

    test_fea = feature_extract(test_pd,is_train = False)
    test_fea = pd.DataFrame(test_fea)
    test_fea_mean = test_fea.groupby(0,as_index = False).mean()
    # remove id column
    test_fea_mean.rename(columns={0:'id'},inplace=True)
    test_x_df = test_fea_mean.drop(columns=['id'])
    test_x_array = np.array(test_x_df)
    test_x = torch.tensor(test_x_array).float()
    test_x = test_x / 255.0 
    return train_x, train_y, test_x

def draw_fig(train_loss_ls,test_loss_ls,train_samstep,test_samstep):
    date=datetime.datetime.today()
    data_str=date.strftime("%Y-%m-%d")
    figure=plt.figure()
    axes1=figure.add_subplot(2,1,1)
    axes2 = figure.add_subplot(2, 1, 2)
    trainloss_x=np.arange(0,len(train_loss_ls))
    testloss_x=np.arange(0,len(test_loss_ls))
    axes1.plot(testloss_x[1:]*test_samstep,test_losses[1:],label="test loss")
    axes2.plot(trainloss_x[1:]*train_ud_samstep, train_losses[1:],label="train loss")
    axes1.set_xlabel("epoch")
    axes1.set_ylabel("test mse")
    axes2.set_xlabel("epoch")
    axes2.set_ylabel("train mse")
    axes1.legend()
    axes2.legend()
    plt.savefig("Deecamp_nn_%s.png"%data_str)

def train():
    pass

if __name__== "__main__":
    # train_data_path = './deecamp_ad/train_8000.csv'
    train_data_path = 'train_data.csv'
    test_data_path = './deecamp_ad/test_2000.csv'
    model_save_path="./model_fcl_cyy_drop.pkl"
    x, y, sub_x = data_loader(train_data_path, test_data_path) # sub_x 待提交数据特征
    test_num=x.size()[0]//5
    train_num = y.size()[0] - test_num
    # pdb.set_trace()
    experiment_i = 0

    # 五折交叉验证数据集划分
    test_x=x[experiment_i*test_num:(experiment_i+1)*test_num]
    test_y=y[experiment_i*test_num:(experiment_i+1)*test_num]
    train_x=torch.cat((x[0:experiment_i*test_num],x[(experiment_i+1)*test_num:]),0)
    train_y=torch.cat((y[0:experiment_i*test_num],y[(experiment_i+1)*test_num:]),0)

    net = Net()
    criterian=nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0005)
    batch_size, maxloop = 64, 30
    test_losses, train_losses = [], []
    train_ud_samstep, test_epoch_samstep = 1, 5
    delta=1 # huber 损失参数
    for epoch in range(maxloop):
        randix=torch.randperm(train_num)    # shuffle 关键操作
        for i in range(train_num//batch_size+1): 
            # print("a",train_num,batch_size,train_num//batch_size+1)
            if i < train_num//batch_size:
                inputs = train_x[randix[i * batch_size:(i+1) * batch_size]]
                labels = train_y[randix[i * batch_size:(i+1) * batch_size]]
            else:
                if i * batch_size < train_num:
                    inputs = train_x[randix[i * batch_size:]]
                    labels = train_y[randix[i * batch_size:]]    

            optimizer.zero_grad()
            # pdb.set_trace()
            outputs=net(inputs)

            # loss=criterian(outputs,labels)
            # huber 损失函数的实现
            if abs(outputs-labels).float().mean()<=delta:
                loss=((outputs-labels)**2).float().mean()
                # print("-"*25,epoch,i,abs(outputs-labels).float().mean().item(),loss.item())
            else:
                # print("-"*50,epoch,i,abs(outputs-labels).float().mean().item(),loss.item())
                loss=delta*abs(outputs-labels).float().mean()-0.5*delta**2

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if (i+1) % train_ud_samstep == 0:
                print("train epoch:%d i:%d loss:%.3f"%(epoch,i,loss))
            # 异常, 记录, 损失
            file=open("."+"/loss_log.txt","a")
            date_in=datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            file.write("date in %s  epoch %s i %s : loss is %s inputs mean is %s y_label %s"%(date_in,epoch,i,loss.item(),inputs.mean().item(),labels.mean().item()))
            file.write("\n")
            file.close()

        # 每个epoch 性能测试 数据少，直接输出，不分小minibatch
        if (epoch+1) % test_epoch_samstep ==0:
            out_test=net(test_x)
            test_loss=criterian(out_test,test_y)
            print("test epoch:%d loss:%.3f" % (epoch + 1, test_loss))
            test_losses.append(test_loss.item())
        
    print("Finished Training")
    torch.save(net,model_save_path)

    print("plot figures")
    draw_fig(train_losses, test_losses, train_ud_samstep, test_epoch_samstep)

    print("calculate submission.csv")
    test_pd = pd.read_csv(test_data_path)
    sub = pd.DataFrame()
    sub['id'] = test_pd['id']
    sub['kpi'] = net(sub_x).detach().numpy()
    sub.to_csv('submission.csv',index=False)




