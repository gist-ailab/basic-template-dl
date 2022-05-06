import os
import pathlib

from torch import set_default_dtype
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

import numpy as np
import json
import subprocess
from multiprocessing import Process
import argparse
import warnings
warnings.filterwarnings('ignore')

import random

def load_json(json_path):
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out


def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=20)
    args = parser.parse_args()

    # Data Configuration
    json_data_path = '../config/base_data.json'
    json_data = load_json(json_data_path)

    # Network Configuration
    json_network_path = '../config/base_network.json'
    json_network = load_json(json_network_path)

    # Train Configuration
    json_train_path = '../config/base_train.json'
    json_train = load_json(json_train_path)

    # Meta Configuration
    json_meta_path = '../config/base_meta.json'
    json_meta = load_json(json_meta_path)

    # Global Option
    train_prop = 1.
    val_prop = 1.

    project_folder = 'AMAL-FREE'
    log = True
    batch_size = 128

    ################################## CFAR100-B50 / B10 ###############################
    # 1. Init - Cifar100
    if args.exp == 1:
        server = 'toast'
        save_dir_init = '/data/sung/checkpoint/inclearn/'
        data_dir = '/data/sung/dataset'

        epoch = 170

        train_prop = 1.
        val_prop = 1. 
        
        batch_size = 128
        mixed_precision = False
        ddp = False
        
        gpus = ['0', '1', '2', '3', '4', '5']
        num_per_gpu = 1
        
        
        # Conditional Options
        network_list = ['resnet18']
        data = ('cifar100', 100)

        comb_list = []
        for n_t in network_list:
            for init_class in [5, 10, 20, 50]:
                for seed in [11, 64, 148]:
                    class_list = list(range(100))
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(class_list)
                    
                    comb_list.append({'train':
                                            {'lr': 0.1,
                                            'optimizer': 'sgd',
                                            'scheduler': 'step_warmup',
                                            'weight_decay': 0.0005,
                                            'step_milestones': [110, 130],
                                            'step_gamma': 0.1,
                                            'warmup_epoch': 10,
                                            
                                            'init': True,
                                            'total_task': -1,
                                            
                                            'class_list': class_list,
                                            "num_init_class": init_class,
                                            "num_new_class": init_class
                                            },
                                    'meta': {'task': 'init_cifar100_B%d_seed%d' %(init_class, seed),
                                            'mode': 'incremental-learning'},
                                    'network': 
                                            {'extractor_type': n_t},
                                    'data':
                                            {'data_type': data[0],
                                            'num_class': data[1],
                                            'seed': seed},
                                    'index': 'init/cifar100_B%d/%s_fc/seed%d' %(init_class, n_t, seed),
                                    })


    # 2. Init - ImageNet100
    elif args.exp == 2:
        server = 'toast'
        save_dir_init = '/data/sung/checkpoint/inclearn/'
        data_dir = '/data/sung/dataset'

        epoch = 130

        train_prop = 1.
        val_prop = 1. 
        
        batch_size = 256
        mixed_precision = False
        ddp = False
        
        gpus = ['4', '5', '6', '7']
        num_per_gpu = 1
        
        
        # Conditional Options
        network_list = ['resnet18']
        data = ('imagenet100', 100)

        comb_list = []
        for n_t in network_list:
            for init_class in [10, 50]:
                for seed in [11, 64, 148]:
                    class_list = list(range(100))
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(class_list)
                    
                    comb_list.append({'train':
                                            {'lr': 0.1,
                                            'optimizer': 'sgd',
                                            'scheduler': 'step_warmup',
                                            'weight_decay': 0.0005,
                                            'step_milestones': [40, 70, 90, 100],
                                            'step_gamma': 0.1,
                                            'warmup_epoch': 10,
                                            
                                            'init': True,
                                            'total_task': -1,
                                            
                                            'class_list': class_list,
                                            "num_init_class": init_class,
                                            "num_new_class": init_class
                                            },
                                    'meta': {'task': 'init_imagenet100_B%d_seed%d' %(init_class, seed),
                                            'mode': 'incremental-learning'},
                                    'network': 
                                            {'extractor_type': n_t},
                                    'data':
                                            {'data_type': data[0],
                                            'num_class': data[1],
                                            'seed': seed},
                                    'index': 'init/imagenet100_B%d/%s_fc/seed%d' %(init_class, n_t, seed),
                                    })


    # 3. Init - ImageNet1000
    elif args.exp == 3:
        server = 'toast'
        save_dir_init = '/data/sung/checkpoint/inclearn/'
        data_dir = '/data/sung/dataset'

        epoch = 100

        train_prop = 1.
        val_prop = 1. 
        
        batch_size = 256
        mixed_precision = False
        ddp = True
        
        gpus = ['4,5', '6,7']
        num_per_gpu = 1
        
        
        # Conditional Options
        network_list = ['resnet18']
        data = ('imagenet', 1000)

        comb_list = []
        for n_t in network_list:
            for init_class in [100, 500, 800]:
                for seed in [11, 64, 148]:
                    class_list = list(range(1000))
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(class_list)
                    
                    comb_list.append({'train':
                                            {'lr': 0.2,
                                            'optimizer': 'sgd',
                                            'scheduler': 'step_warmup',
                                            'weight_decay': 0.0005,
                                            'step_milestones': [40, 70],
                                            'step_gamma': 0.1,
                                            'warmup_epoch': 10,
                                            
                                            'init': True,
                                            'total_task': -1,
                                            
                                            'class_list': class_list,
                                            "num_init_class": init_class,
                                            "num_new_class": init_class
                                            },
                                    'meta': {'task': 'init_imagenet_B%d_seed%d' %(init_class, seed),
                                            'mode': 'incremental-learning'},
                                    'network': 
                                            {'extractor_type': n_t},
                                    'data':
                                            {'data_type': data[0],
                                            'num_class': data[1],
                                            'seed': seed},
                                    'index': 'init/imagenet_B%d/%s_fc/seed%d' %(init_class, n_t, seed),
                                    })    

    
    else:
        raise('Select Proper Experiment Number')

    arr = np.array_split(comb_list, len(gpus))
    arr_dict = {}
    for ix in range(len(gpus)):
        arr_dict[ix] = arr[ix]

    def tr_gpu(comb, ix):
        comb = comb[ix]
        
        global json_data
        global json_network
        global json_train
        global json_meta
        
        global save_dir_init
        for i, comb_ix in enumerate(comb):
            save_dir = os.path.join(save_dir_init, comb_ix['index'])
            os.makedirs(save_dir, exist_ok=True)

            gpu = gpus[ix]

            ## 1. Common Options
            # Modify the data configuration
            json_data['data_dir'] = data_dir

            # Modify the train configuration
            json_train['gpu'] = str(gpu)

            json_train['total_epoch'] = epoch
            json_train['batch_size'] = batch_size

            json_train["mixed_precision"] = mixed_precision

            json_train["train_prop"] = train_prop
            json_train["val_prop"] = val_prop

            json_train["ddp"] = ddp

            # Modify the meta configuration
            json_meta['server'] = str(server)
            json_meta['save_dir'] = str(save_dir)
            json_meta['project_folder'] = project_folder
           
            ## 2. Conditional Options
            for key in comb_ix.keys():
                if key == 'train':
                    module = json_train
                elif key == 'data':
                    module = json_data
                elif key == 'network':
                    module = json_network
                elif key == 'meta':
                    module = json_meta
                elif key == 'index':
                    continue
                else:
                    raise('Select Proper Configure Types')
                
                for key_ in comb_ix[key].keys():
                    module[key_] = comb_ix[key][key_]

                if key == 'train':
                    json_train = module
                elif key == 'data':
                    json_data = module
                elif key == 'network':
                    json_network = module
                elif key == 'meta':
                    json_meta = module
                else:
                    raise('Select Proper Configure Types')
            
                module = None
            
            # Save the Configure
            save_json(json_data, os.path.join(save_dir, 'data.json'))
            save_json(json_network, os.path.join(save_dir, 'network.json'))
            save_json(json_train, os.path.join(save_dir, 'train.json'))
            save_json(json_meta, os.path.join(save_dir, 'meta.json'))
                
            # Run !
            script = 'CUDA_VISIBLE_DEVICES=%s python ../train.py --save_dir %s --log %s' %(str(gpu), save_dir, log)
            subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)