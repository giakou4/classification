# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Tue Nov 16 00:33:26 2021

Convolutional Neural Network Example with TorchEnsemble
Input images are 28 x 28 x 1 (1 channel) from MNIST dataset. Output is mapped
to 10 classes (numbers 0 to 9).

https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/examples/classification_cifar10_cnn.py
"""

import numpy as np
import torch
from torchvision import datasets, transforms
# from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
# from torchensemble.bagging import BaggingClassifier
# from torchensemble.gradient_boosting import GradientBoostingClassifier
# from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
# from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier

# Set the seeds to ensure reproducibility
np.random.seed(1)
torch.manual_seed(1)
   

class Model(torch.nn.Module):
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
        # L4 FC 4 x 4 x 128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
            )
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten them for FC
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.nn.functional.softmax(x, dim=1)
    

def main():
    
    num_epochs = 10
    num_estimators = 10
    use_gpu = True if torch.cuda.is_available() else False
    n_jobs = 12
    
    model = Model()
    ensemble = VotingClassifier(estimator=model, n_estimators=num_estimators, cuda=use_gpu, n_jobs=n_jobs)
    ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=1e-3)
    ensemble.set_scheduler("CosineAnnealingLR", T_max=num_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    ensemble.set_criterion(criterion)
    
    transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
    
    train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
    test_set = datasets.MNIST('./data', download=True, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
    
    ensemble.fit(train_loader, epochs=num_epochs, test_loader=None)
    acc, loss = ensemble.evaluate(test_loader, return_loss=True)
    print('Testing Accuracy: {} | Loss: {}'.format(acc, loss))


if __name__ == "__main__":
    main()
