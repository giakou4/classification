# MNIST Classification

Classification of the MNIST dataset in Pytorch using 3 different approaches:
* Convolutional Neural Networks (CNN)
* Contrastive Learning (CL) framework SimCLR
* Multiple Instance Learning (MIL)


## Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 70,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. It has a training set of 60,000 examples, and a test set of 10,000 examples.

# What is Contrastive Learning
In recent years, a resurgence of work in CL has led to major advances in selfsupervised representation learning. The common idea in these works is the following: pull together an anchor and a “positive” sample in embedding space, and push apart the anchor from many “negative” samples. If no labels are available (unsupervised contrastive learning), a positive pair often consists of data augmentations of the sample, and negative pairs are formed by the anchor and randomly chosen samples from the minibatch. 
<img src="https://raw.githubusercontent.com/HobbitLong/SupContrast/master/figures/teaser.png" alt="Contrastive Learning" style="height: 100px; width:200px;"/>

## What is Multiple Instance Learning
In the classical (binary) supervised learning problem one aims at finding a model that predicts a value of a target variable, `y ∈ {0, 1}`, for a given instance, `x`. In the case of the MIL problem, however, instead of a single instance there is a bag of instances, `X = {x1, . . . , xn}`, that exhibit neither dependency nor ordering among each other. We assume that `n` could vary for different bags. There is also a single binary label `Y` associated with the bag. Furthermore, we assume that individual labels exist for the instances within a bag, i.e., `y1,...,yn` and `yk ∈ {0, 1}`, for `k = 1,..., n`, however, there is no access to those labels and they remain unknown during training.
<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-38617-7_6/MediaObjects/468866_1_En_6_Fig1_HTML.png" alt="Multiple Instance Learning" style="height: 100px; width:200px;"/>

## Requirements
```
torch
torchvision
matplotlib
numpy
```

