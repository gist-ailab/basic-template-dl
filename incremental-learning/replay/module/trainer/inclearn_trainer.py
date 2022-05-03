import numpy as np
import torch
from tqdm import tqdm
import os
from utility.distributed import reduce_tensor
from copy import deepcopy
import torch.distributed as dist
from ray import tune
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(option, rank, epoch, model, criterion, optimizer, multi_gpu, tr_loader, scaler, save_module, neptune, save_folder):
    # GPU setup
    num_gpu = len(option.result['train']['gpu'].split(','))

    # For Log
    mean_loss_cls = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Freeze !
    model.train()
    # TODO: Kill the gradient for old class (if common classifier)

    # Run
    for iter, tr_data in enumerate(tqdm(tr_loader)):
        input, label, _ = tr_data
        input, label = input.to(rank), label.to(rank)

        # Forward
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output, _ = model(input)
                loss_cls = criterion(output, label)
                scaler.scale(loss_cls).backward()
                scaler.step(optimizer)
                scaler.update()
                
        else:
            output, _ = model(input)
            loss_cls = criterion(output, label)
            loss_cls.backward()
            optimizer.step()                

        print(loss_cls)
        if not torch.isfinite(loss_cls):
            print('Break because of NaN loss')
            return 0
        else:
            pass            

        # Empty Un-necessary Memory
        torch.cuda.empty_cache()
        
        # Metrics
        acc_result = accuracy(output, label, topk=(1, 5))
        
        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss_cls += reduce_tensor(loss_cls.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss_cls += loss_cls.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]
        
        del output, loss_cls

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss_cls /= len(tr_loader)
    
    # Saving Network Params
    if multi_gpu:
        model_param = deepcopy(model.module.state_dict())
    else:
        model_param = deepcopy(model.state_dict())

    # Save
    save_module.save_dict['model'] = model_param
    save_module.save_dict['optimizer'] = optimizer.state_dict()
    save_module.save_dict['save_epoch'] = epoch


    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss_cls:%.3f' %(epoch, option.result['train']['total_epoch'], mean_acc1, mean_acc5, mean_loss_cls))
        neptune['result/tr_loss_cls'].log(mean_loss_cls)
        neptune['result/tr_acc1'].log(mean_acc1)
        neptune['result/tr_acc5'].log(mean_acc5)
        neptune['result/epoch'].log(epoch)
        
        # Save
        if epoch % option.result['train']['save_epoch'] == 0:
            torch.save({'model':model_param}, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    if multi_gpu and option.result['train']['ddp']:
        dist.barrier()

    return save_module


def validation(option, rank, epoch, model, criterion, multi_gpu, val_loader, scaler, neptune):
    # GPU
    num_gpu = len(option.result['train']['gpu'].split(','))
        
    # Freeze !
    model.eval()
    
    # For Log
    mean_loss_cls = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    with torch.no_grad():
        for iter, val_data in enumerate(tqdm(val_loader)):                
            input, label, _ = val_data
            input, label = input.to(rank), label.to(rank)

            output, _ = model(input)
            loss_cls = criterion(output, label)
            acc_result = accuracy(output, label, topk=(1, 5))
            
            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_loss_cls += reduce_tensor(loss_cls.data, num_gpu).item()
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_loss_cls += loss_cls.item()
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]
    
    # Remove Un-neccessary Memory
    del output, loss_cls
    torch.cuda.empty_cache()
    
    
    # Train Result
    mean_acc1 /= len(val_loader)
    mean_acc5 /= len(val_loader)
    mean_loss_cls /= len(val_loader)


    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss_cls:%.3f' % (epoch, option.result['train']['total_epoch'], mean_acc1, mean_acc5, mean_loss_cls))
        neptune['result/val_loss_cls'].log(mean_loss_cls)
        neptune['result/val_acc1'].log(mean_acc1)
        neptune['result/val_acc5'].log(mean_acc5)

    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss_cls}
    return result