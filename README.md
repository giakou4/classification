# MNIST Classification

Classification of the MNIST dataset in Pytorch using 3 different approaches:
* Convolutional Neural Networks (CNN)
* Contrastive Learning (CL) framework SimCLR
* Multiple Instance Learning (MIL)


## 1. Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 70,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. It has a training set of 60,000 examples, and a test set of 10,000 examples.


## 2. What is Contrastive Learning
In recent years, a resurgence of work in CL has led to major advances in selfsupervised representation learning. The common idea in these works is the following: pull together an anchor and a “positive” sample in embedding space, and push apart the anchor from many “negative” samples. If no labels are available (unsupervised contrastive learning), a positive pair often consists of data augmentations of the sample, and negative pairs are formed by the anchor and randomly chosen samples from the minibatch.  

<p align="center">
<img src="https://raw.githubusercontent.com/HobbitLong/SupContrast/master/figures/teaser.png" alt="Contrastive Learning" style="height: 400px; width:800px;"/>
</p>

More information on:
* SupContrast - Supervised (check https://arxiv.org/abs/2004.11362 & https://arxiv.org/abs/2002.05709)
* SimCLR - Unsupervised (check https://arxiv.org/abs/2004.11362 & https://arxiv.org/abs/2002.05709) 


## 3. What is Multiple Instance Learning
In the classical (binary) supervised learning problem one aims at finding a model that predicts a value of a target variable, `y ∈ {0, 1}`, for a given instance, `x`. In the case of the MIL problem, however, instead of a single instance there is a bag of instances, `X = {x1, . . . , xn}`, that exhibit neither dependency nor ordering among each other. We assume that `n` could vary for different bags. There is also a single binary label `Y` associated with the bag. Furthermore, we assume that individual labels exist for the instances within a bag, i.e., `y1,...,yn` and `yk ∈ {0, 1}`, for `k = 1,..., n`, however, there is no access to those labels and they remain unknown during training. 

<p align="center">
<img src="https://www.researchgate.net/publication/315925709/figure/fig1/AS:555691916382209@1509498685605/An-illustration-of-the-concept-of-multiple-instance-learning-In-MIL-training-examples.png" alt="Multiple Instance Learning" style="height: 300px; width:600px;"/>
</p>
  
More information on:
* Attention (see https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
* Gated Attention (see https://github.com/AMLab-Amsterdam/AttentionDeepMIL)


## 4. Requirements

```
torch
torchvision
matplotlib
numpy
```

## 5. Support

Reach out to me:
- [giakou4's email](mailto:giakonick98@gmail.com "giakonick98@gmail.com")

## 6. License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/MNIST_classification/LICENSE)