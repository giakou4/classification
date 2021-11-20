# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Tue Nov 16 00:33:26 2021

Multiple Instance Learning Example.
Input images are 28 x 28 x 1 (1 channel) from MNIST dataset. The goal is to 
seperate number 9 among all other numbers.
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor, Attention, BagDataset

   
class Model(torch.nn.Module):
    "Convolutional Neural Network"
    def __init__(self):
        super(Model, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L2 (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
            )
        # L3 (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.2)
            )
        self._to_linear = 4 * 4 * 128

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten them for FC
        return x
    
    
def calculate_accuracy(output, target):
    "Calculates accuracy"
    output = output.data.max(dim=1,keepdim=True)[1]
    output = output == 1.0
    output = torch.flatten(output)
    target = target == 1.0
    target = torch.flatten(target)
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item() 


def training(epoch, model, train_loader, optimizer, criterion):
    "Training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        bag_label = labels[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        data , bag_label = torch.autograd.Variable(data,False), torch.autograd.Variable(bag_label)
        loss = model.calculate_objective(data.float(), bag_label)
        error, _ = model.calculate_classification_error(data.float(), bag_label)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", 1-error)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def validation(epoch, model, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    for batch_idx, (data, labels) in enumerate(valid_loader):
        bag_label = labels[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = torch.autograd.Variable(data,False), torch.autograd.Variable(bag_label)
        loss = model.calculate_objective(data.float(), bag_label)
        error, predicted_label = model.calculate_classification_error(data.float(), bag_label)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", 1-error)
    print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def main():
    
    num_epochs = 10
    use_early_stopping = False
    use_scheduler = True
    
    model = Attention(Model()).cuda() if torch.cuda.is_available() else Attention(Model())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
    
    train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
    valid_set = datasets.MNIST('./data', download=True, train=False, transform=transform)
    
    num_train = len(train_set)
    num_valid = len(valid_set)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=num_train, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=num_valid, shuffle=False)
        
    train_loader_bags = torch.utils.data.DataLoader(BagDataset(
                                                        loader=train_loader,
                                                        dataset_length=num_train,
                                                        target_number=9,
                                                        mean_bag_length=10,
                                                        var_bag_length=2,
                                                        num_bag=100,
                                                        seed=1,
                                                        ),
                                                    batch_size=1,
                                                    shuffle=True)
    valid_loader_bags = torch.utils.data.DataLoader(BagDataset(
                                                        loader=valid_loader,
                                                        dataset_length=num_valid,
                                                        target_number=9,
                                                        mean_bag_length=10,
                                                        var_bag_length=2,
                                                        num_bag=250,
                                                        seed=1,
                                                        ),
                                                    batch_size=1,
                                                    shuffle=False)
    
    train_losses , train_accuracies = [],[]
    valid_losses , valid_accuracies = [],[]
    
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-4)
 
    for epoch in range(1, num_epochs+1):
        
        train_loss, train_accuracy = training(epoch,model,train_loader_bags,optimizer,criterion)
        valid_loss, valid_accuracy = validation(epoch,model,valid_loader_bags,criterion)
        
        if use_scheduler:
            scheduler.step()
            
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
             
        if use_early_stopping: 
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print('Early stopping at epoch', epoch)
                #model.load_state_dict(torch.load('checkpoint.pt'))
                break
     
    plt.plot(range(1,len(train_losses)+1), train_losses, color='b', label = 'training loss')
    plt.plot(range(1,len(valid_losses)+1), valid_losses, color='r', linestyle='dashed', label = 'validation loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
     
    plt.plot(range(1,len(train_accuracies)+1),train_accuracies, color='b', label = 'training accuracy')
    plt.plot(range(1,len(valid_accuracies)+1),valid_accuracies, color='r', linestyle='dashed', label = 'validation accuracy')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Accuracy'), plt.show()

if __name__ == "__main__":
    main()