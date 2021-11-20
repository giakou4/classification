# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Tue Nov 16 00:33:26 2021

Contrastive Learning Example using the framework SimCLR.
Input images are 28 x 28 x 1 (1 channel) from MNIST dataset. Output is mapped
to 10 classes (numbers 0 to 9).
"""

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor
from util import TwoCropTransform, SupCon, SupConLoss, save_model

   
class Encoder(torch.nn.Module):
    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()
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
    

class LinearClassifier(torch.nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 10),
            )

    def forward(self, x):
        x = self.fc(x)
        probs = torch.nn.functional.softmax(x, dim=0)
        return probs


def calculate_accuracy(output, target):
    "Calculates accuracy"
    output = output.data.max(dim=1,keepdim=True)[1]
    output = output == 1.0
    output = torch.flatten(output)
    target = target == 1.0
    target = torch.flatten(target)
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item() 


def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    for batch_idx, (data,labels) in enumerate(contrastive_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        bsz = labels.shape[0]
        features = model(data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if method == 'SupCon':
            loss = criterion(features, labels)
        elif method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.format(method))
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg']
        
def training(epoch, model, classifier, train_loader, optimizer, criterion):
    "Training over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        with torch.no_grad():
            features = model.encoder(data)
        output = classifier(features.float())
        loss = criterion(output, labels) 
        accuracy = calculate_accuracy(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        data.detach()
        labels.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def validation(epoch, model, classifier, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for batch_idx, (data,labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data,labels = data.cuda(), labels.cuda()
            data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
            features = model.encoder(data)
            output = classifier(features.float())
            loss = criterion(output,labels) 
            accuracy = calculate_accuracy(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            data.detach()
            labels.detach()
    print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def main():
    
    num_epochs = 2
    use_early_stopping = False
    use_scheduler = True
    save_file = os.path.join('./results/', 'model.pth')
    if not os.path.isdir('./results/'):
         os.makedirs('./results/')
    
    contrastive_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    train_transform = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    valid_transform = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                       ])
    
    contrastive_set = datasets.MNIST('./data', download=True, train=True, transform=TwoCropTransform(contrastive_transform))
    train_set = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
    valid_set = datasets.MNIST('./data', download=True, train=False, transform=valid_transform)
    
    contrastive_loader = torch.utils.data.DataLoader(contrastive_set, batch_size=64, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)
    
    # Part 1
    encoder = Encoder()
    model = SupCon(encoder, head='mlp', feat_dim=128)
    criterion = SupConLoss(temperature=0.07)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    contrastive_loss, contrastive_lr = [], []
    
    for epoch in range(1, num_epochs+1):
        loss, lr = pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR')
        if use_scheduler:
            scheduler.step()
        contrastive_loss.append(loss)
        contrastive_lr.append(lr)
    
    save_model(model, optimizer, num_epochs, save_file)
    
    plt.plot(range(1,len(contrastive_lr)+1),contrastive_lr, color='b', label = 'learning rate')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Learning Rate'), plt.show()
    
    plt.plot(range(1,len(contrastive_loss)+1),contrastive_loss, color='b', label = 'loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
    
    # Part 2
    model = SupCon(encoder, head='mlp', feat_dim=128)
    classifier = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    
    ckpt = torch.load(save_file, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    train_losses , train_accuracies = [],[]
    valid_losses , valid_accuracies = [],[]
    
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-4)
 
    for epoch in range(1, num_epochs+1):
        
        train_loss, train_accuracy = training(epoch, model, classifier, train_loader, optimizer, criterion)
        valid_loss, valid_accuracy = validation(epoch, model, classifier, valid_loader, criterion)
        
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