"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

DEFAULT_THRESHOLD = 5e-3

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    @staticmethod
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None


class ElementWiseConv2d(nn.Module):
    """Modified conv with masks for weights."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ElementWiseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        self.mask_real.fill_(1.0)

        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)


    def forward(self, input):
        # Mask weights with above mask.
        weight_thresholded = self.mask_real * self.weight
        # Perform conv using modified weight.
        return F.conv2d(input, weight_thresholded, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True):
        super(ElementWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        self.mask_real.fill_(1.0)

        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)


    def forward(self, input):
        # Mask weights with above mask.
        weight_thresholded = self.mask_real * self.weight
        # Get output using modified weight.
        return F.linear(input, weight_thresholded, self.bias)
    
    def _apply(self, fn):
            for module in self.children():
                module._apply(fn)

            for param in self._parameters.values():
                if param is not None:
                    # Variables stored in modules are graph leaves, and we don't
                    # want to create copy nodes, so we have to unpack the data.
                    param.data = fn(param.data)
                    if param._grad is not None:
                        param._grad.data = fn(param._grad.data)

            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)

            self.weight.data = fn(self.weight.data)
            self.bias.data = fn(self.bias.data)