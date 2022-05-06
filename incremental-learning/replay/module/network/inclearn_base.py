import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from .resnet import resnet18, resnet34, resnet50, resnet101

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Wrapper for Incremental Learning (Task == 0)
class Init_Model(nn.Module):
    def __init__(self, option):
        super(Init_Model, self).__init__()
        self.option = option
        
        self.extractor_0 = self.set_extractor(task_id=0)
        self.classifier_0 = self.set_classifier(task_id=0, num_class=self.option.result['train']['num_init_class'])
        
        self.old_class_tot = 0
        self.new_class_tot = self.option.result['train']['num_init_class']
        
        self.exemplar_list = []

        
    def forward(self, image):
        feat = self.extractor_0(image)
        out = self.classifier_0(feat)
        return out, feat
    
    
    def set_extractor(self, task_id):
        if self.option.result['network']['extractor_type'] == 'resnet18':
            extractor = resnet18(dataset=self.option.result['data']['data_type'], start_class=task_id)
            self.num_embed = 512
            
        elif self.option.result['network']['extractor_type'] == 'resnet34':
            extractor = resnet34(dataset=self.option.result['data']['data_type'], start_class=task_id)
            self.num_embed = 512
            
        else:
            raise('Select Proper Extractor Type')
        
        return extractor
    
    
    def set_classifier(self, task_id, num_class, cos=False):
        if cos:
            # TODO: Cosine FC
            pass
        else: 
            classifier = nn.Linear(self.num_embed, num_class)
        return classifier


# Wrapper for Incremental Learning (Task > 0)
class Incremental_Model(nn.Module):
    def __init__(self, option):
        super(Incremental_Model, self).__init__()
        self.option = option
        
        self.current_task = self.option.result['train']['current_task']
        assert (self.current_task > 0)        
        
        self.num_init_class = self.option.result['train']['num_init_class']
        self.num_new_class = self.option.result['train']['num_new_class']


        ## Old Model Settings for Loadindg Pretrained Model
        # Set Extractor
        if self.option.result['train']['common_extractor']:
            self.extractor_0 = self.set_extractor(task_id=0)
            self.extractor_0.requires_grad = False
        else:
            for ix in range(self.current_task):
                setattr(self, 'extractor_%d' %ix, self.set_extractor(task_id=ix))
                getattr(self, 'extractor_%d' %ix).requires_grad = False
        
        
        # Set Classifier
        self.old_class_tot = self.num_init_class + (self.num_new_class * (self.current_task - 1))
        self.new_class_tot = self.old_class_tot + self.num_new_class
        
        if self.option.result['train']['common_classifier']:
            self.classifier_0 = self.set_classifier(task_id=0, num_class=self.old_class_tot)
            self.classifier_0.requires_grad = False
        else:
            for ix in range(self.current_task):
                if ix == 0:
                    num_class = self.num_init_class
                else:
                    num_class = self.num_new_class
                
                setattr(self, 'classifier_%d' %ix, self.set_classifier(task_id=ix, num_class=num_class))
                getattr(self, 'classifier_%d' %ix).requires_grad = False
        
        
        # Exemplar List
        self.exemplar_list = []


    def set_exemplar(self, exemplar_list):
        # TODO: Load Exemplar
        self.exemplar_list = exemplar_list
    
    
    def set_new_task_module(self):
        # Update Extractor
        if self.option.result['train']['common_extractor']:
            # TODO: Select Freeze or Not the Common Extractor
            self.extractor_0.requires_grad = False
        else: 
            setattr(self, 'extractor_%d' %self.current_task, self.set_extractor(task_id=self.current_task))
            getattr(self, 'extractor_%d' %self.current_task).requires_grad = True
            
        # Update Classifier
        if self.option.result['train']['common_classifier']:
            # TODO: move old classifier checkpoint -> new classifier
            self.classifier_0 = self.set_classifier(task_id=self.current_task, num_class=self.new_class_tot)
            self.classifier_0.requires_grad = True
        else:
            setattr(self, 'classifier_%d' %self.current_task, self.set_classifier(task_id=self.current_task, num_class=self.num_new_class))
            getattr(self, 'classifier_%d' %self.current_task).requires_grad = True
    
    
    def set_extractor(self, task_id):
        if self.option.result['network']['extractor_type'] == 'resnet18':
            extractor = resnet18(dataset=self.option.result['data']['data_type'], start_class=task_id)
        elif self.option.result['network']['extractor_type'] == 'resnet34':
            extractor = resnet34(dataset=self.option.result['data']['data_type'], start_class=task_id)
        else:
            raise('Select Proper Extractor Type')

        return extractor
    
    
    def set_classifier(self, task_id, num_class, cos=False):
        # TODO: calculate the num_embed based on the incremental method and model type
        num_embed = 512
        
        if cos:
            # TODO: Cosine FC
            pass
        else: 
            classifier = nn.Linear(num_embed, num_class)
        return classifier


    def forward(self, image):
        common_extractor, common_classifier = self.option.result['train']['common_extractor'], self.option.result['train']['common_classifier']
        
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
            if common_classifier:
                out = self.classifier_0(feat)
                return out, feat
            
            else:
                out_list = []
                for ix in range(self.current_task + 1):
                    out_ix = getattr(self, 'classifier_%d' %ix)(feat)
                    out_list.append(out_ix)
                return out_list, feat
         
                
        else:
            if common_classifier:
                # TODO: concat feat list
                feat = feat_list
                out = self.classifier_0(feat)
                return out, feat_list
        
            else:
                out_list = []
                for ix in range(self.current_task + 1):
                    out_ix = getattr(self, 'classifier_%d' %ix)(feat_list[ix])
                    out_list.append(out_ix)
                return out_list, feat_list
                
            
    def update_exemplar(self, data, m, rank):
        # TODO: Update Exemplar Code
        # Calculate the centers
        if self.option.result['exemplar']['sampling_type'] == 'herding':
            for img, label in ex_loader_imp:
                x = img.to(rank)
                with torch.no_grad():
                    feature_imp = self.model_enc(x)
                    logit =  self.model_fc(feature_imp).detach()
                    feature = F.normalize(feature_imp).detach().cpu().numpy()
                features.append(feature)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

            features = np.concatenate(features, axis=0)
            logits = np.concatenate(logits, axis=0)
            labels = np.concatenate(labels, axis=0)
            class_mean = np.mean(features, axis=0)

            # Find the optimal exemplar set
            exemplar_set = []
            feature_dim = features.shape[1]
            now_class_mean = np.zeros((1, feature_dim))

            for i in range(m):
                # shape：batch_size*512
                x = class_mean - (now_class_mean + features) / (i + 1)
                # shape：batch_size
                x = np.linalg.norm(x, axis=1)
                index = np.argmin(x)
                now_class_mean += features[index]

                if self.option.result['exemplar']['exemplar_type'] == 'data':
                    exemplar_set.append(data[index])
                elif self.option.result['exemplar']['exemplar_type'] == 'logit':
                    exemplar_set.append((torch.tensor(logits[index]).float(), torch.tensor([labels[index]]).long().item()))
                else:
                    raise('Select Proper exemplar type')

        else:
            raise('Select Proper Exemplar Sampling Type')

        # Reduce Old Exemplar Set
        for ix in range(len(self.exemplar_list)):
            self.exemplar_list[ix] = self.exemplar_list[ix][:m]


    def my_hook(self, grad):
        grad_clone = grad.clone()
        grad_clone[:self.old_class_tot] = 0.0
        return grad_clone

    def register_hook(self):
        self.hook = self.model_fc.weight.register_hook(self.my_hook)

    def remove_hook(self):
        self.hook.remove()


if __name__=='__main__':
    pass
