import numpy as np
from glob import glob
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import subprocess
from tqdm import tqdm


def move_file(im1k_dir, im100_dir):
    # Train
    with open('train_100.txt', 'r') as f:
        data = f.readlines()
    
    print('Move Train File')
    for d in tqdm(data):
        file_name = d.strip().split(' ')[0]
        old_path = os.path.join(im1k_dir, file_name)
        new_path = os.path.join(im100_dir, file_name)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
        script = 'cp -r %s %s' %(old_path, new_path)
        subprocess.call(script, shell=True)

    
    # Validation
    with open('val_100.txt', 'r') as f:
        data = f.readlines()

    print('Move Validation File')
    for d in data:
        file_name = d.strip().split(' ')[0]
        old_path = os.path.join(im1k_dir, file_name)
        new_path = os.path.join(im100_dir, file_name)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
        script = 'cp -r %s %s' %(old_path, new_path)
        subprocess.call(script, shell=True)    



if __name__=='__main__':
    imagenet1k_folder = '/data/sung/dataset/imagenet'
    imagenet100_folder = '/data/sung/dataset/imagenet100'
    move_file(imagenet1k_folder, imagenet100_folder)