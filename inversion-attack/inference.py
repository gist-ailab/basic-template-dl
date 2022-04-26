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
    # i = args.data_id
    i = 0
    device = 'cuda:7'

    # data_root = '/home/dataset/imagenet'
    # classifier_path = '/home/personal/shin_sungho/checkpoint/imagenet100/%d/best_model.pt' %i
    data_root = '/data/sung/dataset/imagenet'
    classifier_0_path = '/data/sung/checkpoint/imagenet100/%d/best_model.pt' %i
    
    epoch = 1
    model_path = 'checkpoint/data_%d/generator_%d_%s/epoch_%d.pt' %(i, int(class_cond), class_type, epoch)
    
    # target_list = list(range(100 * i, 100 * (i+1)))
    target_list = list(range(100, 200))
    
    batch_size = 256
    total_epoch = 20
    num_class = 100
    
    if class_cond:
        num_embed = 2048 + 100
    else:
        num_embed = 2048

    # Dataset
    val_dataset = load_data(data_root=data_root, data_type='val')
    val_dataset = IncrementalSet(val_dataset, target_list=target_list, shuffle_label=False, prop=1.0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # ImageNet Options
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

    # Classifier
    classifier_0 = create_model('resnet50', pretrained=False, num_classes=100)
    classifier_0.load_state_dict(torch.load(classifier_0_path, map_location='cpu')[0])
    classifier_0 = classifier_0.to(device)
    classifier_0 = classifier_0.eval()

    # Attach Batch Norm Hook
    batch_features_0 = []
    batch_features_0.append(FeatHook(classifier_0.layer4))

    # Generator
    model = Generator_Split(num_embed)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)

    # Criterion
    criterion_mse = nn.MSELoss()

    # Validation
    val_loss = 0.
    model.eval()
    for image, _ in tqdm(val_loader):
        image = image.to(device)
        
        # Forward Path
        with torch.no_grad():
            output = classifier_0(image)
            
            label = torch.max(output, dim=1)[1]
            output = output.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7])

            if class_cond:
                if class_type == 'output':            
                    latent = torch.cat([batch_features_0[0].r_feature, output], dim=1)
                elif class_type == 'label':
                    label = F.one_hot(label, num_class)
                    label = label.unsqueeze(2).unsqueeze(3).repeat([1,1,7,7]) 
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
    
    os.makedirs('ood_result/data_%d/generator_%d_%s' %(i, int(class_cond), class_type), exist_ok=True)
    plt.savefig('ood_result/data_%d/generator_%d_%s/epoch_%d.png' %(i, int(class_cond), class_type, epoch))
    plt.close(1)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--class_cond', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--class_type', type=str, default='label')
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()
    main(args)