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
from model.inversion import Generator_Split
import torch.nn.functional as F
from tqdm import tqdm
import argparse

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

def main(args):
    # Option
    class_cond = args.class_cond
    class_type = args.class_type
    i = args.data_id
    device = 'cuda'

    # data_root = '/home/dataset/imagenet'
    # classifier_path = '/home/personal/shin_sungho/checkpoint/imagenet100/%d/best_model.pt' %i
    data_root = '/data/sung/dataset/imagenet'
    classifier_path = '/data/sung/checkpoint/imagenet100/%d/best_model.pt' %i
    
    
    target_list = list(range(100 * i, 100 * (i+1)))
    batch_size = 256
    total_epoch = 20
    num_class = 100
    
    if class_cond:
        num_embed = 2048 + 100
    else:
        num_embed = 2048

    # Dataset
    tr_dataset = load_data(data_root=data_root, data_type='train')
    val_dataset = load_data(data_root=data_root, data_type='val')

    tr_dataset = IncrementalSet(tr_dataset, target_list=target_list, shuffle_label=True, prop=1.0)
    val_dataset = IncrementalSet(val_dataset, target_list=target_list, shuffle_label=False, prop=1.0)

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # ImageNet Options
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

    # Classifier
    classifier = create_model('resnet50', pretrained=False, num_classes=100)
    classifier.load_state_dict(torch.load(classifier_path)[0])
    classifier = classifier.to(device)
    classifier = classifier.eval()

    # Attach Batch Norm Hook
    batch_features_0 = []
    batch_features_0.append(FeatHook(classifier.layer4))

    # Generator
    model = Generator_Split(num_embed)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Criterion
    criterion_mse = nn.MSELoss()

    for epoch in range(total_epoch):
        # Train
        model.train()
        tr_loss = 0.
        for image, label in tqdm(tr_loader):
            image = image.to(device)
            label = label.to(device)
            label_new = F.one_hot(label, num_class)
            label_new = label_new.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7])
            
            # Forward Path
            optimizer.zero_grad()
            
            with torch.no_grad():
                output = classifier(image)

            # Class Conditional            
            if class_cond:
                if class_type == 'output':        
                    output_new = output.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7])
                    latent = torch.cat([batch_features_0[0].r_feature, output_new], dim=1)
                elif class_type == 'label': 
                    latent = torch.cat([batch_features_0[0].r_feature, label_new], dim=1)
                else:
                    raise('Select Proper Class Type')
            else:
                latent = batch_features_0[0].r_feature
                
            img_recon = model(latent)
            img_target = (((image * std) + mean) - 0.5) * 2
            loss = criterion_mse(img_recon, img_target)
            
            if class_cond:
                output_new = classifier((((img_recon + 1) / 2) - mean) / std) 

                if class_type == 'output':
                    T = 5
                    alpha = 0.3
                    loss_ce = nn.KLDivLoss()(F.log_softmax(output_new/T, dim=1),
                                           F.softmax(output/T, dim=1)) * (alpha * T * T) + \
                            F.cross_entropy(output_new, label) * (1. - alpha)
                    loss += loss_ce * 0.1

                elif class_type == 'label':
                    loss_ce = nn.CrossEntropyLoss()(output_new, label)
                    loss += loss_ce * 0.1

                else:
                    raise('Select Proper Class Type')
                
            loss.backward()
            optimizer.step()
        
            tr_loss += loss.item()
            
        tr_loss /= len(tr_loader)
        print('Training (%d/%d) -- loss %.2f' %(epoch, total_epoch, tr_loss))
        
        # Validation
        val_loss = 0.
        model.eval()
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            label = F.one_hot(label, num_class)
            label = label.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7])
            
            # Forward Path
            with torch.no_grad():
                output = classifier(image)
                output = output.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7])

                if class_cond:
                    if class_type == 'output':            
                        latent = torch.cat([batch_features_0[0].r_feature, output], dim=1)
                    elif class_type == 'label': 
                        latent = torch.cat([batch_features_0[0].r_feature, label], dim=1)
                    else:
                        raise('Select Proper Class Type')
                else:
                    latent = batch_features_0[0].r_feature
                
                img_recon = model(latent)
            
            img_target = (((image * std) + mean) - 0.5) * 2
            loss = criterion_mse(img_recon, img_target)
            
            val_loss += loss.item()
            
        val_loss /= len(val_loader)
        print('Validation (%d/%d) -- loss %.2f' %(epoch, total_epoch, val_loss))

        # Save Picture
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)

        img_recon = ((img_recon + 1) / 2)[0] * 255
        img_recon = img_recon.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
        axes[0].imshow(img_recon)
            
        img_target = ((img_target + 1) / 2)[0] * 255
        img_target = img_target.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
        axes[1].imshow(img_target)
        plt.title('MSE LOSS : %.3f' %val_loss)
        
        os.makedirs('samples/data_%d/generator_%d_%s' %(i, int(class_cond), class_type), exist_ok=True)
        plt.savefig('samples/data_%d/generator_%d_%s/epoch_%d.png' %(i, int(class_cond), class_type, epoch))
        plt.close(1)
        
        # Save Checkpoint
        os.makedirs('checkpoint/data_%d/generator_%d_%s' %(i, int(class_cond), class_type), exist_ok=True)
        torch.save(model.state_dict(), 'checkpoint/data_%d/generator_%d_%s/epoch_%d.pt' %(i, int(class_cond), class_type, epoch))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--class_cond', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--class_type', type=str, default='label')
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()
    main(args)