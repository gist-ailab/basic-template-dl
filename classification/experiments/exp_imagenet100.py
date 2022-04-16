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

    # Meta Configuration
    json_tune_path = '../config/base_tune.json'
    json_tune = load_json(json_tune_path)

    # Global Option
    train_prop = 1.
    val_prop = 1.

    project_folder = 'AMAL-FREE'
    resume = False
    mixed_precision = True

    ddp = False
    log = True
    
    batch_size = 128

    # Setup Configuration for Each Experiments
    if args.exp == 0:
        server = 'lecun'
        save_dir_init = '/home/personal/shin_sungho/checkpoint/data-free'
        data_dir = '/data/sung/dataset'

        exp_name = 'imp'
        comb_list = []
        epoch = 100

        train_prop = 1.
        val_prop = 1. 
        
        batch_size = 512
        mixed_precision = True
        ddp = False
        
        num_per_gpu = 1
        
        gpus = ['2,3', '4,5']
        
        # Conditional Options
        network_list = ['resnet50']
        data_type_list = [('imagenet', 1000)]
        
        class_list = sorted(list(range(1000)))

        for data in data_type_list:
            for n_t in network_list:
                for ix in range(2):
                    target_list = class_list[(100 * ix):(100 *(ix+1))]
                    data_num = len(target_list)

                    comb_list.append({'train': 
                                            {'lr': 1.2,
                                            'expand_channels': 0,
                                            'scheduler': 'cycle',
                                            'target_list': target_list,
                                            'pretrained_imagenet': False
                                            },
                                    'network': 
                                            {'network_type': n_t},
                                    'data':
                                            {'data_type': data[0],
                                            'num_class': data_num},
                                    'index': '%s/%d' %(n_t, ix),
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
        global json_tune
        
        global save_dir_init, exp_name
        for i, comb_ix in enumerate(comb):
            save_dir = os.path.join(save_dir_init, exp_name, comb_ix['index'])
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

            json_train["resume"] = resume

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
            save_json(json_tune, os.path.join(save_dir, 'tune.json'))
                
            # Run !
            script = 'CUDA_VISIBLE_DEVICES=%s python ../main.py --save_dir %s --log %s' %(str(gpu), save_dir, log)
            subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)