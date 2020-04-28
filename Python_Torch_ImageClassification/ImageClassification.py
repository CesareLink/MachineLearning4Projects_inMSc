# -*- coding: utf-8 -*-
#搭建针对图片的CNN:CNN4ImageClassification_SimpleEdition
#_____________________________________________________________________________________
#模块导入:module import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
print('PyTorch Version',torch.__version__)

#_____________________________________________________________________________________
#导入数据:data load
#data download
mnist_data = datasets.MNIST('./mnist_data', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),]))
print(len(mnist_data))
mnist_data[223][0].shape
#dataset setting

batch_size = 32
#train dataloader
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True, pin_memory=True)

#test dataloader
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size, shuffle=True, pin_memory=True)

#_____________________________________________________________________________________
#定义一个简单的CNN网络:CNN_Net_Structure Defination
class Net(nn.Module):
    """
    
    """
    #Initialization
    def __init__(self):
        super(Net, self).__init__()
        #first conv layer with input dim 1, output dim 20, conv size 5,
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  #from 28*28 to 24*24
        #first conv layer with input dim 20, output dim 50, conv size 5,1
        self.conv2 = nn.Conv2d(20, 50, 5, 1) #from 12*12 to 8*8
        #first fc layer with input 4*4*50, and nodes 500
        #(4*4 as the size of datamap,50 as the feature map number)
        self.fc1 = nn.Linear(4*4*50, 500)
        #second fc layer with input 500, and nodes 10
        self.fc2 = nn.Linear(500, 10)

    #Forward Pass
    def forward(self, x):
        #the forward pass of input data (1*28*28)
        x = F.relu(self.conv1(x)) #from 28*28 to 24*24
        x = F.max_pool2d(x, 2, 2) #from 24*24 to 12*12
        x = F.relu(self.conv2(x)) #from 12*12 to 8*8
        x = F.max_pool2d(x, 2, 2) #from 8*8 to 4*4
        x = x.view(-1, 4*4*50) #resize the tensor = flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
#_____________________________________________________________________________________
#定义训练过程和测试过程: Defination of training and testing process
#用于被主程序调用以实现训练和测试过程，内容应包含给定模型，输入，数据，迭代次数
#As the function for training and testing, including model, input, data, interval
#Training process
def train(model, train_loader, optimizer, epoch, log_interval=100):
    #this is a training function defination
    model.train()     
    #读取train_loader读取的batch内的数据:load the data in batch from train_loader
    for idx, (data, target) in enumerate(train_loader):
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
                epoch, idx * len(data), len(train_loader.dataset), 
                100. * idx / len(train_loader), loss.item()
            ))

#test process
def test(model, test_loader):
    #defination of test_loss and correct count
    test_loss = 0
    correct = 0
    #this is a training function defination
    model.eval()     
    #读取train_loader读取的batch内的数据:load the data in batch from train_loader
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            #从数据中获得预测值:obtain preditions from data model
            output = model(data) #batch_size *10
            #计算总的损失和收益
            pred = output.argmax(dim=1, keepdim=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#_____________________________________________________________________________________
#set the hyperparameters 
lr = 0.01
momentum = 0.5
epochs = 2
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
print(model)

for epoch in range(1, epochs + 1):
    train(model, train_loader=train_dataloader, optimizer=optimizer, epoch=epoch)
    test(model, test_loader=test_dataloader)

save_model = True
if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
    
