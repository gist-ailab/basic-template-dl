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




def entropy(preds, use_softmax=True):
    if use_softmax:
        preds = torch.nn.Softmax(dim=-1)(preds)

    logp = torch.log(preds + 1e-5)
    entropy = torch.sum(-preds * logp, dim=-1)
    return entropy
    
    
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
from dataset.dataset import load_data, IncrementalSet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if __name__=='__main__':
    device = torch.device('cuda:5')
    data_root = '/data/sung/dataset/imagenet'
    batch_size = 128
        
    # Dataset
    val_dataset = load_data(data_root=data_root, data_type='val')
    val_dataset_0 = IncrementalSet(val_dataset, target_list=list(range(100)), shuffle_label=False, prop=1.0)
    val_loader_0 = DataLoader(val_dataset_0, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    val_dataset_1 = IncrementalSet(val_dataset, target_list=list(range(100, 200)), shuffle_label=False, prop=1.0)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    ## Load Two Classifiers (Regularization)
    # ImageNet (0 ~ 99 classifier)
    classifier_0_path = '/data/sung/checkpoint/imagenet100/0/best_model.pt'
    classifier_0 = create_model('resnet50', pretrained=False, num_classes=100)
    classifier_0.load_state_dict(torch.load(classifier_0_path)[0])
    classifier_0 = classifier_0.to(device)
    classifier_0 = classifier_0.eval()
    
    
    # ImageNet (100 ~ 200 classifier)
    classifier_1_path = '/data/sung/checkpoint/imagenet100/1/best_model.pt'
    classifier_1 = create_model('resnet50', pretrained=False, num_classes=100)
    classifier_1.load_state_dict(torch.load(classifier_1_path)[0])
    classifier_1 = classifier_1.to(device)
    classifier_1 = classifier_1.eval()
    
    correct_0, correct_1, wrong_0, wrong_1 = 0., 0., 0., 0.
    total = 0.
    for img, label in tqdm(val_loader_0):
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            output_0 = classifier_0(img)
            output_1 = classifier_1(img)
        
        
        out_pred = torch.cat([torch.max(output_0, dim=1)[1].view(-1,1), 100 + torch.max(output_1, dim=1)[1].view(-1,1)], dim=1)
        out_osr = ((entropy(output_0) - entropy(output_1)) > 0.).long()

        ID_pred_result = torch.max(output_0, dim=1)[1] == label
        
        correct_0 += len(ID_pred_result[out_osr == 0]) - torch.sum(ID_pred_result[out_osr == 0])
        correct_1 += torch.sum(ID_pred_result[out_osr == 0])
        
        wrong_0 += len(ID_pred_result[out_osr == 1])  - torch.sum(ID_pred_result[out_osr == 1])
        wrong_1 += torch.sum(ID_pred_result[out_osr == 1])
    
    print('MODEL 0')
    print(correct_0, correct_1, wrong_0, wrong_1)
    

    correct_0, correct_1, wrong_0, wrong_1 = 0., 0., 0., 0.
    total = 0.
    for img, label in tqdm(val_loader_1):
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            output_0 = classifier_0(img)
            output_1 = classifier_1(img)
        
        
        out_pred = torch.cat([torch.max(output_0, dim=1)[1].view(-1,1), 100 + torch.max(output_1, dim=1)[1].view(-1,1)], dim=1)
        out_osr = ((entropy(output_0) - entropy(output_1)) > 0.).long()

        ID_pred_result = torch.max(output_1, dim=1)[1] == label
        
        correct_0 += len(ID_pred_result[out_osr == 1]) - torch.sum(ID_pred_result[out_osr == 1])
        correct_1 += torch.sum(ID_pred_result[out_osr == 1])
        
        wrong_0 += len(ID_pred_result[out_osr == 0])  - torch.sum(ID_pred_result[out_osr == 0])
        wrong_1 += torch.sum(ID_pred_result[out_osr == 0])
    
    print('MODEL 1')
    print(correct_0, correct_1, wrong_0, wrong_1)