# -*- coding: utf-8 -*-
#CNN4EEG_Classification_SimpleEdition
#_____________________________________________________________________________________
#模块导入:module import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io as scio
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
print('PyTorch Version',torch.__version__)
    
#_____________________________________________________________________________________
#导入数据:data load
class EEGDataset():
    #init
    def __init__(self, file_name):
        #load the data
        #data is set as (train_data,train_label,test_data and test_label)
        data = scio.loadmat(file_name)
        self.data = data
        eeg_data_train = data['train_x']
        eeg_label_train = data['train_y']
        eeg_data_test = data['test_x']
        eeg_label_test = data['test_y']
        self.label = eeg_label_train + eeg_label_test
        return eeg_data_train, eeg_label_train, eeg_data_test, eeg_label_test
     
    #getitem
    def __getitem__(self):
        data1, label1, data2, label2 = self.eeg_data
        #reshape the data and label into 
        data1 = data1.reshape([8000, 30, 200, 1])
        data2 = data2.reshape([2400, 30, 200, 1])
        label1 = label1.reshape([8000,3]) #one-hot label
        label2 = label2.reshape([2400,3])
        train_data = [(data1, label1)]
        test_data = [(data2, label2)]
        return train_data, test_data
    #the number of total sample
    def __len__(self):
        return len(self.label)


#dataset setting
batch_size = 32
#train dataloader
train_dataloader = torch.utils.data.DataLoader(
    EEGDataset[0], train=True, transform=transforms.Compose([transforms.ToTensor(), 
    ]), batch_size=batch_size, shuffle=True, pin_memory=True)

#test dataloader
test_dataloader = torch.utils.data.DataLoader(
    EEGDataset[1], train=False, transform=transforms.Compose([transforms.ToTensor(), 
    ]), batch_size=batch_size, shuffle=True, pin_memory=True)

#_____________________________________________________________________________________
#定义一个简单的CNN网络:CNN_Net_Structure Defination
class Net(nn.Module):
    """
    without any input, a simple module of CNN, output the tensor calculated.
    2 Convolutional layers and 2 fc layers
    """
    #Initialization
    def __init__(self):
        super(Net, self).__init__()
        #first conv layer with input dim 1, output dim 20, conv size 3,
        self.conv1 = nn.Conv2d(1, 20, (1, 3), 1)  #from 30*200 to 30*198
        #first conv layer with input dim 20, output dim 50, conv size 3,1
        self.conv2 = nn.Conv2d(20, 50, (1, 3), 1) #from 30*198 to 30*196
        #first fc layer with input 50, and nodes 80
        self.fc1 = nn.Linear(50, 80)
        #second fc layer with input 80, and nodes 3
        self.fc2 = nn.Linear(80, 3)

    #Forward Pass
    def forward(self, x):
        #the forward pass of input data (1*30*200)
        x = F.relu(self.conv1(x)) #from 30*200 to 30*198
        x = F.relu(self.conv2(x)) #from 30*198 to 30*196
        x = F.avg_pool2d(x, 1, 196) #channel averagepooling, from 30*196 to 30*1
        x = x.view(-1, 30*50) #resize the tensor = flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
#_____________________________________________________________________________________
#定义训练过程和测试过程: Defination of training and testing process
#用于被主程序调用以实现训练和测试过程，内容应包含给定模型，输入，数据，迭代次数
#As the function for training and testing, including model, input, data, interval
#Training process
def train(model, train_dataloader, optimizer, epoch, log_interval=100):
    #this is a training function defination
    model.train()     
    #读取train_loader读取的batch内的数据:load the data in batch from train_loader
    for idx, (data, target) in enumerate(train_dataloader):
        #从数据中获得预测值:obtain preditions from data model
        pred = model(data) #batch_size *10
        #计算损失函数NLL Loss:calculate loss function NLL
        loss = F.nll_loss(pred, target)
        #SGD
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                epoch, idx * len(data), len(train_dataloader.dataset), 
                100. * idx / len(train_dataloader), loss.item()
            ))

#test process
def test(model, test_dataloader):
    #defination of test_loss and correct count
    test_loss = 0
    correct = 0
    #this is a training function defination
    model.eval()     
    #读取train_loader读取的batch内的数据:load the data in batch from train_loader
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            #从数据中获得预测值:obtain preditions from data model
            output = model(data) #batch_size *10
            #计算总的损失和收益
            pred = output.argmax(dim=1, keepdim=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

#_____________________________________________________________________________________
#set the hyperparameters 
lr = 0.01
momentum = 0.5
epochs = 2
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
print(model)

for epoch in range(1, epochs + 1):
    train(model, train_dataloader, optimizer, epoch)
    test(model, test_dataloader)

save_model = True
if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
    
