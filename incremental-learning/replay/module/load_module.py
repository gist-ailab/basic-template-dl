import torch
import torch.nn as nn
from utility.warmup_scheduler import GradualWarmupScheduler
from .network.inclearn_base import Init_Model, Incremental_Model


def load_model(option):
    current_task = option.result['train']['current_task']
    if current_task == 0:
        model = Init_Model(option)
    else:
        model = Incremental_Model(option)
    return model


def load_optimizer(option, model):
    weight_decay = option.result['train']['weight_decay']
    
    # Classifier
    param_cls = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer    
    if option.result['train']['optimizer'] == 'sgd':
        optim_cls = torch.optim.SGD(param_cls, lr=option.result['train']['lr'], momentum=0.9, weight_decay=weight_decay)
        
    elif option.result['train']['optimizer'] == 'adam':
        optim_cls = torch.optim.Adam(param_cls, lr=option.result['train']['lr'])
    
    else:
        raise('Select Proper Optimizer')
    
    return optim_cls


def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(option.result['train']['total_epoch']/3), int(option.result['train']['total_epoch']*2/3)])
        
    elif option.result['train']['scheduler'] == 'anealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
        
    elif option.result['train']['scheduler'] == 'cycle':
        pct_start = 4 / option.result['train']['total_epoch']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=option.result['train']['lr'], pct_start=pct_start, div_factor=1e4, final_div_factor=1e8,
                                                steps_per_epoch=1, epochs=(option.result['train']['total_epoch']), anneal_strategy='linear')
        
    elif option.result['train']['scheduler'] == 'anealing_warmup':
        scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=int(option.result['train']['total_epoch'] / 20), after_scheduler=scheduler_cls)
        
    else:
        raise('Select Proper Scheduler')
    
    return scheduler


def load_loss(option):
    criterion = nn.CrossEntropyLoss()
    return criterion

