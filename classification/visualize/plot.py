import torch.nn as nn
from numpy import dot
from numpy.linalg import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calc_cosine_similarity(model):
    prototype = model.fc.weight.cpu().detach().numpy()
    class_norm = np.linalg.norm(prototype, axis=1).reshape(-1, 1)
    prototype_norm = prototype / class_norm

    similarity = np.matmul(prototype_norm, prototype_norm.T)
    return similarity

def plot_similarity(corr, class_dict):
    plt.figure()
    
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    plt.savefig('imp.png')
    


class attention_manager(object):
    def __init__(self, model, multi_gpu):        
        self.multi_gpu = multi_gpu
        
        self.attention = []
        self.handler = []

        self.model = model
        
        if multi_gpu:
            self.register_hook(self.module.model)
        else:
            self.register_hook(self.model)
            

    def register_hook(self, model):
        def get_attention_features(_, inputs, outputs):
            self.attention.append(outputs)
        
        for name, layer in model._modules.items():
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.register_hook(layer)
            else:
                for name, layer2 in layer._modules.items():
                    if name == 'attention':
                        handle = layer2.register_forward_hook(get_attention_features)
                        self.handler.append(handle)
        
    def get_attention(self, input):
        # Forward Model
        self.attention = []
        out = self.model(input)
        return self.attention        
    
    def remove_handler(self):
        for handler in self.handler:
            handler.remove()
        
    def plot_attention(self, features):
        pass
    

class cam_manager(object):
    def __init__(self, model, target, multi_gpu):        
        self.multi_gpu = multi_gpu
        
        self.features = []
        self.gradients = []
        self.handler = []

        self.model = model
        self.target = target
        if multi_gpu:
            self.register_hook(self.module.model)
        else:
            self.register_hook(self.model)
            
    def register_hook(self, model):
        def get_features(_, inputs, outputs):
            self.features.append(outputs[0])
        
        def get_gradients(_, inputs, outputs):
            self.gradients.append(outputs[0].detach())
    
        handle_f = model._modules[self.target][-1].register_forward_hook(get_features)
        self.handler.append(handle_f)
        
        handle_g = model._modules[self.target][-1].register_backward_hook(get_gradients)
        self.handler.append(handle_g)
        
    def get_gradcam(self, input):
        # Forward Model
        self.features, self.gradients = [], []
        out = self.model(input)
        return self.features, self.gradients
    
    def remove_handler(self):
        for handler in self.handler:
            handler.remove()
            
            
if __name__=='__main__':
    import timm
    model = timm.create_model('resnet18', pretrained=True)
    corr = calc_cosine_similarity(model)
    plot_similarity(corr, None)