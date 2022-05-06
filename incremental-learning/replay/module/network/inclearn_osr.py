import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from .inclearn_base import Incremental_Model
from .osr_module import entropy, mls, msp


# Wrapper for Incremental Learning (Task > 0)
class OSR_Model(Incremental_Model):
    def __init__(self, option):
        super(OSR_Model, self).__init__(option)
        self.inclearn_method = self.option.result['train']['inclearn_method']
        self.option.result['train']['common_classifier'] = False
        

    def set_new_task_module(self):
        # Update Extractor
        if self.option.result['train']['common_extractor']:
            self.extractor_0.requires_grad = False
        else: 
            setattr(self, 'extractor_%d' %self.current_task, self.set_extractor(task_id=self.current_task))
            getattr(self, 'extractor_%d' %self.current_task).requires_grad = True
            
        # Update Classifier
        setattr(self, 'classifier_%d' %self.current_task, self.set_classifier(task_id=self.current_task, num_class=self.num_new_class))
        getattr(self, 'classifier_%d' %self.current_task).requires_grad = True
    
    
    
    def set_classifier(self, task_id, num_class, cos=False):
        # TODO: calculate the num_embed based on the incremental method and model type
        num_embed = 512
        
        if cos:
            # TODO: Cosine FC
            pass
        else: 
            classifier = nn.Linear(num_embed, num_class)

        return classifier


    def osr_selector(self, out_list, feat_list):
        B = out_list[0].size(0)
        
        # Concat the Task Results
        output = []
        for ix, out in enumerate(out_list):
            if ix == 0:
                start_ix = 0
                end_ix = self.option.result['train']['num_init_class']
            else:
                start_ix = self.option.result['train']['num_init_class'] + self.option.result['train']['num_new_class'] * (ix - 1)
                end_ix = start_ix + self.option.result['train']['num_new_class']
                
            out_base = torch.zeros((B, len(self.option.result['train']['whole_class_list']))).to(out.device)
            out_base[:, start_ix:end_ix] = out
            output.append(out_base.view(B, 1, -1))
        
        output = torch.cat(output, dim=1)
        
        # OSR for task discrimination
        if self.inclearn_method == 'entropy':
            osr_result = entropy(output) # B x T
            _, task_idx = torch.min(osr_result, dim=-1)
        elif self.inclearn_method == 'mls':
            osr_result = mls(output)
            _, task_idx = torch.max(osr_result, dim=-1)
        elif self.inclearn_method == 'msp':
            osr_result = msp(output)
            _, task_idx = torch.max(osr_result, dim=-1)
        else:
            raise('Select Proper Inclearn Method')
        
        output = output[list(range(B)), task_idx]
        return output
    
    
    def forward(self, image, train=True):
        common_extractor = self.option.result['train']['common_extractor']
        
        if train:
            # Forward Extractor
            if common_extractor:
                ix = 0
                feat = self.extractor_0(image)
                out = self.classifier_0(feat)
            else:
                ix = self.current_task
                feat = getattr(self, 'extractor_%d' %ix)(image)
                out = getattr(self, 'classifier_%d' %ix)(feat)
            
            return out, feat
                                
        else:
            # Forward Extractor
            if common_extractor:
                feat = self.extractor_0(image)
            else:
                feat_list = []
                for ix in range(self.current_task + 1):
                    feat_ix = getattr(self, 'extractor_%d' %ix)(image)
                    feat_list.append(feat_ix)
            
            
            # Forward Classifier
            if common_extractor:
                out_list = []
                for ix in range(self.current_task + 1):
                    out_ix = getattr(self, 'classifier_%d' %ix)(feat)
                    out_list.append(out_ix)
                
                out = self.osr_selector(out_list, feat)
                return out, feat
            
            else:
                out_list = []
                for ix in range(self.current_task + 1):
                    out_ix = getattr(self, 'classifier_%d' %ix)(feat_list[ix])
                    out_list.append(out_ix)

                out = self.osr_selector(out_list, feat_list)
                return out, feat_list
                
            
if __name__=='__main__':
    pass
