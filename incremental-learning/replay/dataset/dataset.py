from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random
from glob import glob
from PIL import Image

def load_cifar10(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR10(root=option.result['data']['data_dir'], train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=option.result['data']['data_dir'], train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset


def load_cifar100(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR100(root=option.result['data']['data_dir'], train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=option.result['data']['data_dir'], train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset


def load_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    resize= 224

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(resize * 256 / 224)),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_imagenet100(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    resize= 224

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(resize * 256 / 224)),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_data(option, data_type='train'):
    if option.result['data']['data_type'] == 'imagenet':
        tr_d, val_d = load_imagenet(option)
    elif option.result['data']['data_type'] == 'imagenet100':
        tr_d, val_d = load_imagenet100(option)
    elif option.result['data']['data_type'] == 'cifar10':
        tr_d, val_d = load_cifar10(option)
    elif option.result['data']['data_type'] == 'cifar100':
        tr_d, val_d = load_cifar100(option)
    else:
        raise('select appropriate dataset')

    if data_type == 'train':
        return tr_d
    else:
        return val_d


class Base_Folder(Dataset):
    def __init__(self, option, folder, transform=None):
        self.option = option
        self.class_list = sorted(os.listdir(folder))

        self.targets = []
        self.samples = []
        
        self.class_dict = {}
        for id, cls_name in enumerate(self.class_list):
            self.class_dict[id] = cls_name
            
            image_list_ix = glob(os.path.join(folder, cls_name, '*.JPEG'))
            self.samples += image_list_ix
            self.targets += [id] * len(image_list_ix)

        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            
        label = torch.tensor(self.targets[index]).long()
        return image, label


class IncrementalSet(Dataset):
    def __init__(self, dataset, exemplar_list, old_class_num, target_list, train, shuffle_label=False, prop=1.):
        self.dataset = dataset
        self.dataset_label = np.array(self.dataset.targets)
        
        # Select Target Index
        self.target_index = []
        for ix in target_list:
            ix_index = np.where(self.dataset_label == ix)[0]

            np.random.seed(100)
            select_num = int(len(ix_index) * prop)
            ix_index = np.random.choice(ix_index, select_num, replace=False)
            self.target_index.append(ix_index)

        self.target_index = np.concatenate(self.target_index, axis=0)
        
        
        # For Matching Class ID sequentially (0, 1, ... N)
        self.target_dict = {}
        self.old_class_num = old_class_num
        for ix, target in enumerate(target_list):
            if train:
                self.target_dict[target] = old_class_num + ix
            else:
                self.target_dict[target] = ix
                
        
        # Exermplar Dataset
        self.exemplary = exemplar_list
        self.index_list = list(range(len(self.target_index) + len(self.exemplary)))


        # Shuffle Index
        if shuffle_label:
            shuffle(self.index_list)
        self.index_list = np.array(self.index_list)


    def __len__(self):
        return len(self.index_list)
    

    def __getitem__(self, index):
        index = self.index_list[index]

        if index < len(self.target_index):
            image, label = self.dataset.__getitem__(self.target_index[index])
            label = self.target_dict[label]
        else:
            image, label = self.exemplary[index - len(self.target_index)]
        
        if label < self.old_class_num:
            task = 0
        else:
            task = 1
        return image, label, task
    
    
class transform_module():
    def __init__(self, option):
        if 'cifar' in option.result['data']['data_type']:
            mu = np.array([0.5071,  0.4866,  0.4409])
            std = np.array([0.2009,  0.1984,  0.2023])
        elif option.result['data']['data_type'] == 'imagenet':
            mu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        elif option.result['data']['data_type'] == 'food101':
            mu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        else:
            raise ('select appropriate dataset')

        self.mu = mu
        self.std = std

    def normalize(self, img):
        img = torch.squeeze(img)
        img = transforms.Normalize(self.mu, self.std)(img)
        img = torch.unsqueeze(img, dim=0)
        return img

    def un_normalize(self, img):
        transform_un_normalize = transforms.Compose([transforms.Normalize((0,0,0), 1/self.std),
                                                     transforms.Normalize(-self.mu, (1,1,1))])
        img = torch.squeeze(img)
        img = transform_un_normalize(img)
        img = torch.unsqueeze(img, dim=0)
        return img