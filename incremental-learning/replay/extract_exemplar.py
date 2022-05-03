import json
import numpy as np
import argparse
import os
import neptune.new as neptune
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.trainer import init_trainer
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from dataset.dataset import load_data, IncrementalSet

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import pathlib
import random
    

def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, option, resume, save_folder, log, master_port):
    # Basic Options
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    total_epoch = option.result['train']['total_epoch']


    # GPU Configuration
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

        if log:
            mode = 'async'
        else:
            mode = 'debug'

        monitoring_hardware = True
        if resume and option.result['meta']['neptune_id'] is not None:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=monitoring_hardware,
                               capture_stderr=monitoring_hardware,
                               capture_hardware_metrics=monitoring_hardware,
                               run = option.result['meta']['neptune_id'],
                               mode = mode
                               )
        else:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=monitoring_hardware,
                               capture_stderr=monitoring_hardware,
                               capture_hardware_metrics=monitoring_hardware,
                               mode = mode
                               )

        cfg = option.result
        for key in cfg.keys():
            for key_ in cfg[key].keys():
                cfg_name = 'config/%s/%s' %(key, key_)
                run[cfg_name] = cfg[key][key_]
    else:
        run = None


    # Load Model
    model = load_model(option)
    criterion = load_loss(option)
    save_module = train_module(total_epoch, criterion, multi_gpu)

    if resume:
        save_module.import_module(resume_path)
        
        # Load Model
        model.load_state_dict(save_module.save_dict['model'])

        # Load Epoch
        if save_module.save_dict['save_epoch'] == (int(option.result['train']['total_epoch']) - 1):
            return None


    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu, master_port)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model = apply_gradient_allreduce(model)

        criterion.to(rank)

    else:
        if multi_gpu:
            model = nn.DataParallel(model).to(rank)
        else:
            model = model.to(rank)
            

    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])


    # Dataset and DataLoader
    tr_dataset = load_data(option, data_type='train')
    val_dataset = load_data(option, data_type='val')


    # Partial Training by adding self.targets to the custom dataset (CUB, Stanford_Cars, ...)
    if option.result['train']['target_list'] is None:
        target_list = list(range(0, option.result['data']['num_class']))
    else:
        target_list = option.result['train']['target_list']
        
        
    # TODO: Add Whole List for Total Validation
    whole_list = list(map(int, option.result['train']['whole_class_list'].split(',')))
    

    # TODO: Load Exemplar
    if option.result['train']['num_exemplar'] > 0 and option.result['train']['current_task'] > 0:
        exemplar_list = []
    else:
        exemplar_list = []
    
    
    # Start Class for New-task
    if multi_gpu:
        start = model.module.old_class_tot
    else:
        start = model.old_class_tot
    
    tr_dataset = IncrementalSet(tr_dataset,  exemplar_list=exemplar_list, start=start, target_list=target_list, shuffle_label=True, prop=option.result['train']['train_prop'])
    val_dataset = IncrementalSet(val_dataset, exemplar_list=[], start=start, target_list=whole_list, shuffle_label=False, prop=option.result['train']['val_prop'])


    # Data Loader
    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['train']['num_workers'], pin_memory=pin_memory,
                                                  sampler=tr_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['train']['num_workers'], pin_memory=pin_memory,
                                                  sampler=val_sampler)

    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])


    # Optimizer and Scheduler
    optimizer = load_optimizer(option, model)
    
    if resume:
        # Load Optimizer
        optimizer.load_state_dict(save_module.save_dict['optimizer'])

    if option.result['train']['scheduler'] is not None:
        scheduler = load_scheduler(option, optimizer)
        
        if resume:
            scheduler.load_state_dict(save_module.save_dict['scheduler'])
    else:
        scheduler = None


    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None        
    
    
    # Run
    for epoch in range(save_module.init_epoch, save_module.total_epoch):
        # Scheduler with Warm-up
        if scheduler is not None:
            if option.result['train']['scheduler'] == 'anealing':
                scheduler.step(epoch)
            elif option.result['train']['scheduler'] == 'cycle':
                pass
            else:
                scheduler.step()
                    
            save_module.save_dict['scheduler'] = scheduler.state_dict()


        # Train
        save_module = init_trainer.train(option, rank, epoch, model, criterion, optimizer, multi_gpu, \
                                          tr_loader, scaler, save_module, run, save_folder)
        
        
        # Evaluation
        result = init_trainer.validation(option, rank, epoch, model, criterion, multi_gpu, val_loader, scaler, run)
            
            
        # Log Learning Rate
        if run is not None:
            for param_group in optimizer.param_groups:
                run['debug/current_lr'].log(param_group['lr'])
                

        # Cyclic Scheduler
        if scheduler is not None:
            if option.result['train']['scheduler'] == 'cycle':
                scheduler.step()
            else:
                pass                    
            save_module.save_dict['scheduler'] = scheduler.state_dict()


        # Save the last-epoch module
        if (rank == 0) or (rank == 'cuda'):
            save_module_path = os.path.join(save_folder, 'last_dict.pt')
            save_module.export_module(save_module_path)

            save_config_path = os.path.join(save_folder, 'last_config.json')
            option.export_config(save_config_path)


        # Early Stopping
        if result is not None:
            if multi_gpu:
                current_model = deepcopy(model.module.state_dict())
            else:
                current_model = deepcopy(model.state_dict())

            if option.result['train']['early_loss']:
                early(result['val_loss'], current_model, result)
            else:
                early(-result['acc1'], current_model, result)

            if early.early_stop == True:
                break
        

    # Save Best Model
    if result is not None:
        if (rank == 0) or (rank == 'cuda'):
            torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))

    if ddp:
        cleanup()


    # TODO: Select and Update Exemplar for New Task
    if option.result['train']['num_exemplar'] > 0:
        if multi_gpu:
            exemplar_list = model
        else:
            exemplar_list = model

        np.save(exemplar_list, os.path.join(save_folder, 'exemplar.npy'))

    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/data/sung/checkpoint')
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=False)
    args = parser.parse_args()

    
    # Configure
    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)
    
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()
    option.get_config_meta()

    option.result['train']['save_folder'] = save_folder
    option.result['train']['task'] = 'incremental-learning'


    # Resume Configuration
    resume = option.result['train']['resume']
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    config_path = os.path.join(save_folder, 'last_config.json')


    # BASE FOLDER
    option.result['train']['base_folder'] = str(pathlib.Path(__file__).parent.resolve())


    # Resume
    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
        else:
            gpu = option.result['train']['gpu']

            option = config(save_folder)
            option.import_config(config_path)

            option.result['train']['gpu'] = gpu


    # Data Directory
    if not resume:
        option.result['data']['data_dir'] = os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'])


    # Num Class
    option.result['data']['num_class'] = option.result['train']['num_init_class'] + (option.result['train']['num_new_class'] * option.result['train']['current_task'])


    # GPU
    num_gpu = len(option.result['train']['gpu'].split(','))
    option.result['train']['gpu'] = ','.join([str(ix) for ix in range(num_gpu)])

    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    master_port = str(random.randint(1,1000))


    if ddp:
        mp.spawn(main, args=(option, resume, save_folder, args.log, master_port, ), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder, args.log, master_port)
        
    exit()