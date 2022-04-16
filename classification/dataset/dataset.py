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

def load_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if option.result['train']['resize'] is not None:
        resize = option.result['train']['resize']
    else:
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
    def __init__(self, dataset, target_list, shuffle_label=False, prop=1.):
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
        for ix, target in enumerate(target_list):
            self.target_dict[target] = ix
        self.index_list = list(range(len(self.target_index)))

        # Shuffle
        self.shuffle = shuffle_label

        random.seed(100)
        if self.shuffle:
            shuffle(self.index_list)
        self.index_list = np.array(self.index_list)

    def get_image_class(self, label):
        self.target_label_index = np.where(self.dataset_label == label)[0]
        return [self.dataset_exemplar.__getitem__(index) for index in self.target_label_index]
    
    def select_dataset(self, index_list):
        self.index_list = self.index_list[index_list]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        index = self.index_list[index]
        image, label = self.dataset.__getitem__(self.target_index[index])
        label = torch.Tensor([self.target_dict[int(label)]]).long().item()
        return image, label