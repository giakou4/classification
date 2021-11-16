import numpy as np
import torch
from torchvision import datasets, transforms


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, loader, dataset_length, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.loader = loader
        self.r = np.random.RandomState(seed)
        self.dataset_length = dataset_length # 60.000 for train MNIST
        self.loader = loader
        
        self.bag_list, self.labels_list = self._create_bags()

    def _create_bags(self):
        for (batch_data, batch_labels) in self.loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.dataset_length, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bag_list[index]
        label = [max(self.labels_list[index]), self.labels_list[index]]
        return bag, label
    
    
if __name__ == "__main__":

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

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader_bags):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader_bags),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(valid_loader_bags):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(valid_loader_bags),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))