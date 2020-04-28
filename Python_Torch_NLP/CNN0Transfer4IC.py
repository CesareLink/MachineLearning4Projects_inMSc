# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:06:36 2020
CNN Model for Image Classificaiton, Transfer
@author: Cesare
"""
#______________________________________________________________________________
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import time
import os
import copy
print('Torchvision Version:', torchvision.__version__)
#______________________________________________________________________________
#Hyperparameter setting
#setting the name of data
data_dir = './hymenoptera_data'
#settint the name of model
model_name = 'resnet'
#number of classes in the dataset
num_classes = 2
#batch size for training
batch_size =32
#number of epochs to train
num_epochs = 15
#Flag for decision whether we should update the parameters
feature_extract = True

#______________________________________________________________________________
#Model Function Defination
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    #setting time axis
    since = time.time()
    #setting the dictionary for memory the history of acc
    val_acc_history = []
    #setting the dictionary for the final weights of the model
    best_model_wts = copy.deepcopy(model.state_dict())
    #setting the best acc variable
    best_acc = 0
    
    #cycle for training epochs
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        
        #cycle for every epoch training process
        for phase in ['train', 'val']:
            #the loss of each training batch
            running_loss = 0
            running_corrects = 0
            #wheter the process is train or validation
            if phase == 'train':
                #training process-parameter update
                model.train()
            else:
                #evaluation process-parameter don's update
                model.eval()
            #the data from phase= training or validation set    
            for inputs, labels in dataloaders[phase]:
                #if train, the autograd is available
                with torch.autograd.set_grad_enabled(phase=="train"):
                    #the result is the model calculated results
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                #.max means taking the max number in horizental direction
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    #general setting
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                #calculate the batch loss
                running_loss += loss.item() * inputs.size(0)
                #calculate the batch corrects
                running_corrects += torch.sum(preds.view(-1)
                                              == labels.view(-1)).item()
            #calculate the loss and corrects of whole epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            #print the result       
            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                #save the best model weights
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
            
        print()
    #calculate the time consuming
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, 
                                               time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))
    #load the model weights from training
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

#______________________________________________________________________________
#whether the model need gradient descent
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
#______________________________________________________________________________
#ResNet model structure defination
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    #resnet transfer
    if model_name == "resnet":
        #transfer the finetunned model
        model_ft = models.resnet18(pretrained=use_pretrained)
        #transfer the function for banning the gradient descent
        set_parameter_requires_grad(model_ft, feature_extract)
        #.in_features is the number of inputs for the linear layer
        num_ftrs = model_ft.fc.in_features
        #setting the fully-connected layer
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #setting the inputsize of model
        input_size = 224
    #return the output of model_ft and the input size of model_ft    
    return model_ft, input_size

#activate the model function
#'this is in main function'
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

print(model_ft)

#______________________________________________________________________________
#Data Collection
#data collection
#'this is in main function'
#load the data
all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
#set dataloader
loader = torch.utils.data.DataLoader(all_imgs, batch_size=batch_size, shuffle=True, num_workers=4)

#batch and iterator for dataloader
img = next(iter(loader))[0]

#______________________________________________________________________________
#Plot the image
#transform tensor to image data
unloader = transforms.ToPILImage()  # reconvert into PIL image

#plot the image
plt.ion()

#the defination of image presentation
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
#drow the imamge
plt.figure()
imshow(img[31], title='Image')

#______________________________________________________________________________
#Initializing Datasets and Dataloaders
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
            data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
        batch_size=batch_size, shuffle=True,
        num_workers=4) for x in ['train', 'val']}

#
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, ohist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

#______________________________________________________________________________
#plot
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


