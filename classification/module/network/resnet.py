import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn.modules.batchnorm import BatchNorm2d
from .attention import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Attn_Loc(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(Attn_Loc, self).__init__()
        self.inplane = in_channel
        self.reduction_ratio = reduction_ratio

    def forward(self, x):
        return x, []

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='naive'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.attention_type = attention_type

        if attention_type == 'naive':
            self.attention = None
            self.attention_downsample = None
        elif attention_type == 'CBAM':
            self.attention = CBAM(planes, 16)
            self.attention_downsample = None
        elif attention_type == 'SE':
            self.attention = SELayer(planes, 16)
            self.attention_downsample = None
        else:
            kernel_size = 7
            if self.downsample is not None:
                self.attention_downsample = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(1)
                )
            else:
                self.attention_downsample = None
                
            self.attention = nn.Sequential(*[nn.Conv2d(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False),
                                            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
                                            nn.Sigmoid()])
            
    def forward(self, x):
        if self.attention_type not in ['naive', 'CBAM', 'SE']:
            x, attn = x[0], x[1]
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Calculation Attention
        if self.attention_type not in ['naive', 'CBAM', 'SE']: # -> global
            if self.attention_downsample is not None:
                attn = self.attention_downsample(attn)

            attn = self.attention(attn)
            out = out * attn
        else:
            if self.attention is not None:
                out, _ = self.attention(out)
        
        # Skip Connection            
        out += residual
        out = self.relu(out)
        
        if self.attention_type not in ['naive', 'CBAM', 'SE']: # -> global
            return (out, attn)
        else:
            return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='naive'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention_type = attention_type

        if attention_type == 'naive':
            self.attention = None
            self.attention_downsample = None
        elif attention_type == 'CBAM':
            self.attention = CBAM(planes * 4, 16)
            self.attention_downsample = None
        elif attention_type == 'SE':
            self.attention = SELayer(planes * 4, 16)
            self.attention_downsample = None
        else:
            kernel_size = 7
            if self.downsample is not None:
                self.attention_downsample = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(1)
                )
            else:
                self.attention_downsample = None
                
            self.attention = nn.Sequential(*[nn.Conv2d(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False),
                                            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
                                            nn.Sigmoid()])


    def forward(self, x):
        if self.attention_type not in ['naive', 'CBAM', 'SE']:
            x, attn = x[0], x[1]
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # Calculation Attention
        if self.attention_type not in ['naive', 'CBAM', 'SE']: # -> global
            if self.attention_downsample is not None:
                attn = self.attention_downsample(attn)

            attn = self.attention(attn)
            out = out * attn
        else:
            if self.attention is not None:
                out, _ = self.attention(out)
        
        # Skip Connection            
        out += residual
        out = self.relu(out)
        
        if self.attention_type not in ['naive', 'CBAM', 'SE']: # -> global
            return (out, attn)
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, option, block, layers,  network_type, num_classes):
        self.option = option
        self.attention_type = option.result['train']['attention_type']

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type

        # different model config_raw between ImageNet and CIFAR
        if network_type == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Attention
        if self.attention_type in ['naive', 'CBAM', 'SE']:
            self.stem_attention = None
        else:
            self.stem_attention = stem_attention(global_attention=self.option.result['train']['global_attention'])
        
        # Block        
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_type=self.attention_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_type=self.attention_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "imagenet":
            x = self.maxpool(x)

        # Layer 1
        if self.stem_attention is not None:
            attn = self.stem_attention(x)
        else:
            attn = None
        
        if attn is not None:
            x, attn = self.layer1((x, attn))
            x, attn = self.layer2((x, attn))
            x, attn = self.layer3((x, attn))
            x, attn = self.layer4((x, attn))
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # Pooling
        x_ = self.avgpool(x)
        x_ = x_.view(x_.size(0), -1)
        x_out = self.fc(x_)

        return x_out


def ResidualNet(option, network_type, depth, num_classes):
    assert network_type in ["imagenet", "cifar10", "cifar100", "tiny_imagenet"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(option, BasicBlock, [2, 2, 2, 2], network_type, num_classes)

    elif depth == 34:
        model = ResNet(option, BasicBlock, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 50:
        model = ResNet(option, Bottleneck, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 101:
        model = ResNet(option, Bottleneck, [3, 4, 23, 3], network_type, num_classes)
    return model


