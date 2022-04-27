import torch

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import PIL.Image
import torch

import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from timm import create_model


class BatchHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class FeatHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        self.r_feature = output
        # must have no output

    def close(self):
        self.hook.remove()

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

def load_imagenet(data_root):
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

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(data_root, 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    return tr_dataset, val_dataset


if __name__=='__main__':
    device = torch.device('cuda:2')
    first_bn_multiplier = 1.
    
    ## Load Two Classifiers (Regularization)
    # ImageNet (0 ~ 99 classifier)
    classifier_0_path = '/home/sung/checkpoint/imp/0/best_model.pt'
    classifier_0 = create_model('resnet50', pretrained=False, num_classes=100)
    classifier_0.load_state_dict(torch.load(classifier_0_path)[0])
    classifier_0 = classifier_0.to(device)
    classifier_0 = classifier_0.eval()
    
    # ImageNet (100 ~ 200 classifier)
    classifier_1_path = '/home/sung/checkpoint/imp/1/best_model.pt'
    classifier_1 = create_model('resnet50', pretrained=False, num_classes=100)
    classifier_1.load_state_dict(torch.load(classifier_1_path)[0])
    classifier_1 = classifier_1.to(device)
    classifier_1 = classifier_1.eval()
    
    # Attach Batch Norm Hook
    batch_features_0 = []
    batch_features_1 = []
    
    # for module in classifier_0.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         batch_features_0.append(BatchHook(module))
    
    # for module in classifier_1.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         batch_features_1.append(BatchHook(module))
    
    batch_features_0.append(FeatHook(classifier_0.layer1))
    batch_features_1.append(FeatHook(classifier_1.layer1))
    
    batch_features_0.append(FeatHook(classifier_0.layer2))
    batch_features_1.append(FeatHook(classifier_1.layer2))
    
    batch_features_0.append(FeatHook(classifier_0.layer3))
    batch_features_1.append(FeatHook(classifier_1.layer3))
    
    batch_features_0.append(FeatHook(classifier_0.layer4))
    batch_features_1.append(FeatHook(classifier_1.layer4))
    
    
    # ImageNet Options
    _, val_dataset = load_imagenet(data_root='/home/sung/dataset/imagenet')    
    
    for i in range(6000,8000):
        # First Run
        classifier_0.zero_grad()
        classifier_1.zero_grad()
        
        img, label = val_dataset.__getitem__(i)
        img = torch.unsqueeze(img, dim=0).to(device)
        output_0 = classifier_0(img)
        output_1 = classifier_1(img)
            
        target_feat_0, target_feat_1 = [], []
        for i in range(4):
            target_feat_0.append(deepcopy(batch_features_0[i].r_feature.cpu().detach()).to(device))
            target_feat_1.append(deepcopy(batch_features_1[i].r_feature.cpu().detach()).to(device))
                
        # Iter
        w0 = torch.randn((1, 3, 224, 224), requires_grad=True, device=device, dtype=torch.float)
        w1 = torch.randn((1, 3, 224, 224), requires_grad=True, device=device, dtype=torch.float)
        optim_w = torch.optim.Adam([w0, w1], lr=1e-3)
        
        T = 10
        for _ in range(100):
            optim_w.zero_grad()
            
            student_0 = classifier_0(w0)
            student_1 = classifier_1(w1)
            
            loss_mse = 0.
            for i in range(4):
                loss_mse += nn.MSELoss()(batch_features_0[i].r_feature, target_feat_0[i])
                loss_mse += nn.MSELoss()(batch_features_1[i].r_feature, target_feat_1[i])
                
            loss_kl = nn.KLDivLoss()(F.log_softmax(student_0/T, dim=1), F.softmax(output_0/T, dim=1)) + \
                      nn.KLDivLoss()(F.log_softmax(student_1/T, dim=1), F.softmax(output_1/T, dim=1))

            loss = loss_kl + loss_mse
            loss.backward(retain_graph=True)
            optim_w.step()
            
            out1 = nn.MSELoss()(w0, img)
            out2 = nn.MSELoss()(w1, img)
            print(out1, out2)            
            
        print(label)
        print(out1, out2)            
        exit()
        