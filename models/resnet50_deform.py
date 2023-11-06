from typing import List

import torch
import torchvision.ops
import torch.nn as nn

from .resnet50 import ResNet50, default_transformation

deform_conv_option = {
    # first conv, layer1, layer2, layer3, layer4
    'full' : [0, 4, 5, 6, 7],
    #layer1
    'layer1' : [4],
    # layer2
    'layer2' : [5],
    # layer3
    'layer3' : [6],
    # layer4
    'layer4' : [7]
}

class ResNet50_DF(ResNet50):
    def __init__(self, 
                 model_name: str, 
                 num_classes: int, 
                 weights: str = None, 
                 transformation=default_transformation, 
                 feature_extraction: bool = False, 
                 tuned: str = 'full', 
                 pooling: str = 'avgpooling',
                 deform_conv = 'layer4'):
        super().__init__(model_name, num_classes, weights, transformation, feature_extraction, tuned, pooling)
        self.deform_conv_idx = deform_conv_option[deform_conv]
        layers = list(self.encoder.children())
        
        
        for idx in self.deform_conv_idx:
            layer = layers[idx]
            if idx < 4:
                layers[idx] = DeformableConv2d(
                    in_channels=layers[idx].in_channels,
                    out_channels=layers[idx].out_channels,
                    kernel_size=layers[idx].kernel_size,
                    stride=layers[idx].stride,
                    padding=layers[idx].padding,
                    dilation=layers[idx].dilation,
                    bias=layers[idx].bias
                )
            else:
                # Hard coding implementation of replacing all the conv in bottleneck to deform conv
                # except the conv in downsample
                for i, l in enumerate(layer):
                    #l.conv1 = DeformableConv2d(in_channels=l.conv1.in_channels, out_channels=l.conv1.out_channels, kernel_size=l.conv1.kernel_size, stride=l.conv1.stride, padding=l.conv1.padding, dilation=l.conv1.dilation, bias=l.conv1.bias)
                    l.conv2 = DeformableConv2d(in_channels=l.conv2.in_channels, out_channels=l.conv2.out_channels, kernel_size=l.conv2.kernel_size, stride=l.conv2.stride, padding=l.conv2.padding, dilation=l.conv2.dilation, bias=l.conv2.bias)
                    #l.conv3 = DeformableConv2d(in_channels=l.conv3.in_channels, out_channels=l.conv3.out_channels, kernel_size=l.conv3.kernel_size, stride=l.conv3.stride, padding=l.conv3.padding, dilation=l.conv3.dilation, bias=l.conv3.bias)
            if(idx < self.off_layers):
                print("Warning : You are freezing the layers of deformable conv, make sure you are finetuning. Otherwise, the performance may be bad")
                for p in l.parameters():
                    p.requires_grad = False
        
        self.encoder = nn.Sequential(*layers)
    


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)
        
        self.weight = nn.Parameter(regular_conv.weight.data)
        if bias:
            self.bias = nn.Parameter(regular_conv.bias.data)
        else:
            self.bias = None
        del regular_conv

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x