# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Tue Nov 16 00:33:26 2021

Convolutional Neural Network Example using Ensemble
Input images are 28 x 28 x 1 (1 channel) from MNIST dataset. Output is mapped
to 10 classes (numbers 0 to 9).

Ensemble:
Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004, July). Ensemble
selection from libraries of models. In Proceedings of the twenty-first
international conference on Machine learning (p. 18).

Credits:
https://github.com/DmitryBorisenko/ensemble_tutorial/blob/master/MNIST%20Ensembles.ipynb
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from util import EarlyStopping, MetricMonitor
from util import ensemble_selector, cross_entropy, ens_accuracy


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
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data , labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        output = model(data.float())
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


def validation(epoch, model, valid_loader, criterion):
    "Validation over an epoch"
    metric_monitor = MetricMonitor()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data,labels = data.cuda(), labels.cuda()
            data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
            output = model(data.float())
            loss = criterion(output,labels) 
            accuracy = calculate_accuracy(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            data.detach()
            labels.detach()
    print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']
    
    
def plot_model_performance(metrics):   
    # Separate dataframes for losses and accuracies
    metrics_loss = metrics.filter(like="loss").stack().reset_index()
    metrics_loss.columns = ["model", "val/test", "loss"]
    
    metrics_acc = metrics.filter(like="acc").stack().reset_index()
    metrics_acc.columns = ["model", "val/test", "acc"]
    
    # Plot losses and accuracies
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.barplot(x="model", y="loss", hue="val/test", data=metrics_loss,
                alpha=0.75, saturation=0.90, palette=["#1f77b4", "#ff7f0e"],
                ax=ax[0])
    sns.barplot(x="model", y="acc", hue="val/test", data=metrics_acc,
                alpha=0.75, saturation=0.90, palette=["#1f77b4", "#ff7f0e"],
                ax=ax[1])
    
    ax[0].set_ylim(metrics_loss["loss"].min() - 1e-2,
                   metrics_loss["loss"].max() + 1e-2)
    ax[1].set_ylim(metrics_acc["acc"].min()-3e-3,
                   metrics_acc["acc"].max()+3e-3)
    
    ax[0].set_title("Loss", fontsize=17)
    ax[1].set_title("Accuracy", fontsize=17)
    
    for x in ax:
        x.xaxis.set_tick_params(rotation=0, labelsize=15)
        x.yaxis.set_tick_params(rotation=0, labelsize=15)
        x.set_xlabel("Model", visible=True, fontsize=15)
        x.set_ylabel("", visible=False)
    
        handles, labels = x.get_legend_handles_labels()
        x.legend(handles=handles, labels=labels, fontsize=15)
    
    fig.tight_layout(w_pad=5)    


def plot_ensemble_performance(ensemble_metric, ens_metric_val_avg, ensemble_metric_test, ens_metric_test_avg, metric_name):
    '''
    Plot the ensemble losses/accuracies as a funciton of the number of 
    iterations for the validation and test sets on the left and right panels 
    respectively.
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=False)
    ax[0].plot(ensemble_metric, color="#1f77b4", lw=2.75, label="ensemble " + metric_name)
    ax[0].plot(pd.Series(ensemble_metric[0], ensemble_metric.index),
               color="k", lw=1.75, ls="--", dashes=(5, 5),
               label="baseline 1: best model")
    ax[0].plot(pd.Series(ens_metric_val_avg, ensemble_metric.index),
               color="r", lw=1.75, ls="--", dashes=(5, 5),
               label="baseline 2: average of all models")
    
    ax[1].plot(ensemble_metric_test, color="#1f77b4", lw=2.75, label="ensemble " + metric_name)
    ax[1].plot(pd.Series(ensemble_metric_test[0], ensemble_metric_test.index),
               color="k", lw=1.75, ls="--", dashes=(5, 5),
               label="baseline 1: best model on validation set")
    ax[1].plot(pd.Series(ens_metric_test_avg, ensemble_metric.index),
               color="r", lw=1.75, ls="--", dashes=(5, 5),
               label="baseline 2: average of all models")
    
    ax[0].set_title("Validation " + metric_name, fontsize=17)
    ax[1].set_title("Test " + metric_name, fontsize=17)
    
    for x in ax:
        x.margins(x=0.0)
        x.set_xlabel("Optimization Step", fontsize=15, visible=True)
        x.set_ylabel("", fontsize=15, visible=False)
        x.yaxis.set_tick_params(labelsize=15)
        x.xaxis.set_tick_params(labelsize=15)
        x.legend(loc="upper right", bbox_to_anchor=(1, 0.92),
                 frameon=True, edgecolor="k", fancybox=False,
                 framealpha=0.7, shadow=False, ncol=1, fontsize=15)
    fig.tight_layout(w_pad=3.14)
    
    
def plot_ensemble_opt_steps(model_weights):
    '''
    Plot ensemble weights as a function of ensemble optimization steps with 
    lighter hues corresponding to lower average weights over all iterations.
    '''
    # Locate non-zero weights and sort models by their average weight
    weights_to_plot = model_weights.loc[:, (model_weights != 0).any()]
    weights_to_plot = weights_to_plot[weights_to_plot.mean().sort_values(ascending=False).index]
    
    # A palette corresponding to the number of models with non-zero weights
    palette = sns.cubehelix_palette(weights_to_plot.shape[1], reverse=True)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    weights_to_plot.plot(kind="bar", stacked=True, color=palette, ax=ax, alpha=0.85)
    
    ax.margins(x=0.0)
    ax.set_xlabel("Optimization Step", fontsize=15, visible=True)
    ax.set_ylabel("Ensemble Weight", fontsize=15, visible=True)
    ax.yaxis.set_tick_params(rotation=0, labelsize=15)
    ax.xaxis.set_tick_params(rotation=0, labelsize=15)
    ax.legend(loc="best", bbox_to_anchor=(1, 0.92),
              frameon=True, edgecolor="k", fancybox=False,
              framealpha=0.7, shadow=False, ncol=1, fontsize=15)
    fig.tight_layout()
    

def calculate_ensemble_performance(model_weights, 
                                   y_hats_test, y_true_one_hot_test,
                                   y_hats_valid, y_true_one_hot_val):
    '''
    1. Calculates loss & accuracy of weighted ensemble for validation &
    testing set  (ens_loss_valid_avg, ens_loss_test_avg, ens_acc_valid_avg,
                  ens_acc_test_avg)
    2. Calculates the loss & accuracy of the ensemble for the testing set
    (ensemble_loss_test, ensemble_acc_test)
    '''
    
    # Compute test loss & accuracy for each iteration of the ensemble
    ensemble_loss_test = []
    ensemble_acc_test = []
    for _, row in model_weights.iterrows():
        tmp_y_hat = np.array([y_hats_test[model_name] * weight for model_name, weight in row.items()]).sum(axis=0)
        ensemble_loss_test.append(cross_entropy(tmp_y_hat, y_true_one_hot_test))     
        ensemble_acc_test.append(ens_accuracy(tmp_y_hat, y_true_one_hot_test))
    ensemble_loss_test = pd.Series(ensemble_loss_test)
    ensemble_acc_test = pd.Series(ensemble_acc_test)
        
    # Compute loss of the equally weighted ensemble for validation & testing set
    y_valid_avg = np.array([_y for m, _y in y_hats_valid.items()]).mean(axis=0)
    y_test_avg = np.array([_y for m, _y in y_hats_test.items()]).mean(axis=0)
    ens_loss_valid_avg = cross_entropy(y_valid_avg, y_true_one_hot_val)
    ens_loss_test_avg = cross_entropy(y_test_avg, y_true_one_hot_test)
    
    # Compute accuracy of the equally weighted ensemble for validation & testing set
    ens_acc_valid_avg = ens_accuracy(y_valid_avg, y_true_one_hot_val)
    ens_acc_test_avg = ens_accuracy(y_test_avg, y_true_one_hot_test)
    
    return ens_loss_valid_avg, ens_loss_test_avg, ensemble_loss_test,\
           ens_acc_valid_avg,  ens_acc_test_avg, ensemble_acc_test
  
    
def main():
    
    #%% Parameters
    num_models = 5
    num_epochs = 3
    num_iter_ensemble = 10
    num_classes = 10
    valid_split = 0.15
    use_scheduler = True
    model_names = ["M" + str(m) for m in range(num_models)]
    save_file = './results/cnn_ensemble/'
    if not os.path.isdir(save_file):
         os.makedirs(save_file)
    
    #%% Prepare Dataset (plit to training-validation-testing)
    transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
    
    # Load the data
    train_set = datasets.MNIST('./data/', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data/', train=False, download=True, transform=transform)
    
    # Prepare the training and validation samples
    all_indices = list(range(len(train_set)))
    np.random.shuffle(all_indices)
    split = int(np.floor(valid_split * len(train_set)))
    train_idx = all_indices[split:]
    valid_idx = all_indices[:split]
    
    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, sampler=SubsetRandomSampler(indices=train_idx))
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=len(valid_idx), sampler=SubsetRandomSampler(indices=valid_idx))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    
    # Since both validation and test use full batch, fetch the features and targets as tensors
    for batch in valid_loader:
        x_valid, y_valid = batch[0], batch[1]
    for batch in test_loader:
        x_test, y_test = batch[0], batch[1]
        
    #%% Loss function is negative log-likelihood/cross-entropy
    criterion = torch.nn.CrossEntropyLoss()
    
    #%% Train a pool of ensemble candidates
    for model_name in model_names:
    
        # Define the model
        print('\nModel ' + str(model_name))
        model = Model().cuda() if torch.cuda.is_available() else Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []
                
        for epoch in range(num_epochs):
    
            train_loss, train_accuracy = training(epoch,model,train_loader,optimizer,criterion)
            valid_loss, valid_accuracy = validation(epoch,model,valid_loader,criterion)
            
            if use_scheduler:
                scheduler.step()
    
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            # Save the checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": pd.DataFrame(
                    {"train_losses": train_losses, "train_accuracies": train_accuracies,
                     "valid_losses": valid_losses, "valid_accuracies": valid_accuracies}).astype(float),
                "train_loss": train_losses[-1],
                "train_accuracy": train_accuracies[-1],
                "valid_loss": valid_losses[-1],
                "valid_accuracy": valid_accuracies[-1]
                },
                save_file + model_name + "_epoch_" + str(epoch) + ".p")
            
    #%% For each model pick the checkpoint with the lowest validation loss, then:
    trained_models, metrics, y_hats_valid, y_hats_test = {}, {}, {}, {}
    
    for model_name in model_names:
        # Load the last checkpoint
        last_checkpoint = torch.load(save_file + model_name + "_epoch_" + str(num_epochs-1) + ".p")
    
        # Find the best checkpoint by validation loss
        best_by_valid_loss = last_checkpoint["history"].sort_values("valid_losses").index[0]
        best_checkpoint = torch.load(save_file + model_name + "_epoch_" + str(best_by_valid_loss) + ".p")
    
        # Restore the best checkpoint
        model = Model()
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()
    
        # Compute predictions on the validation and test sets
        y_hat_valid = model(x_valid)
        y_hat_test = model(x_test)
        y_hats_valid[model_name] = y_hat_valid.softmax(dim=1).detach().numpy()
        y_hats_test[model_name] = y_hat_test.softmax(dim=1).detach().numpy()
    
        # Compute metrics for test sets
        test_loss = criterion(y_hat_test, y_test).item()
        test_accuracy = calculate_accuracy(y_hat_test, y_test)

        # Store the outputs
        trained_models[model_name] = model
        metrics[model_name] = {
            "valid_loss": best_checkpoint["valid_loss"],
            "valid_accuracy": best_checkpoint["valid_accuracy"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
            }
    
    # Convert the metrics dict to a dataframe
    metrics = pd.DataFrame(metrics).T.astype(float)
    
    # Performance of the individual models first
    plot_model_performance(metrics)
    print('\n\n')
        
    #%% Ensemble minimizing loss    
    
    # One-hot encoded validation & testing labels
    y_true_one_hot_val = np.eye(num_classes)[y_valid.numpy()]
    y_true_one_hot_test = np.eye(num_classes)[y_test.numpy()]
    
    # Ensemble selector: Minimizing Loss
    ensemble_loss, model_weights = ensemble_selector(
        loss_function=cross_entropy, 
        y_hats=y_hats_valid, y_true=y_true_one_hot_val, 
        init_size=1, replacement=True, max_iter=num_iter_ensemble
        )
    
    # Plot ensemble weights
    plot_ensemble_opt_steps(model_weights)
    
    # Calculate results of ensemble and weighted ensemble
    ens_loss_valid_avg, ens_loss_test_avg, ensemble_loss_test,\
    ens_acc_valid_avg,  ens_acc_test_avg, ensemble_acc_test = \
           calculate_ensemble_performance(model_weights,  y_hats_test, y_true_one_hot_test, y_hats_valid, y_true_one_hot_val)
    print('Ensemble (minimize loss) results at testing set: Loss: {} | Accuracy: {}'.format(ensemble_loss_test.iloc[-1], ensemble_acc_test.iloc[-1]))
    
    # Plot the ensemble losses over iterations for validation & testing sets
    plot_ensemble_performance(ensemble_loss, ens_loss_valid_avg, ensemble_loss_test, ens_loss_test_avg, 'Loss')

    #%% Ensemble maximizing accuracy
    
    # Ensemble selector: Maximizing Accuracy => Minimizing (-Accuracy)
    ensemble_acc, model_weights_acc = ensemble_selector(
        loss_function=lambda p, t: -ens_accuracy(p, t),  # - for minimization
        y_hats=y_hats_valid, y_true=y_true_one_hot_val,
        init_size=1, replacement=True, max_iter=num_iter_ensemble
        )
    ensemble_acc = -ensemble_acc  # back to positive domain
    
    # Plot ensemble weights
    plot_ensemble_opt_steps(model_weights_acc)
    
    # Calculate results of ensemble and weighted ensemble
    ens_loss_valid_avg, ens_loss_test_avg, ensemble_loss_test,\
    ens_acc_valid_avg,  ens_acc_test_avg, ensemble_acc_test = \
           calculate_ensemble_performance(model_weights, y_hats_test, y_true_one_hot_test, y_hats_valid, y_true_one_hot_val)
    print('Ensemble (maximize accuracy) results at testing set: Loss: {} | Accuracy: {}'.format(ensemble_loss_test.iloc[-1], ensemble_acc_test.iloc[-1]))
    
    # Plot the ensemble accuracy over iterations for validation & testing sets
    plot_ensemble_performance(ensemble_acc, ens_acc_valid_avg, ensemble_acc_test, ens_acc_test_avg, 'Accuracy')
    

if __name__ == "__main__":
    main()