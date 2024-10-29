import torch.nn.modules.conv as conv
import torch.nn as nn

class Projector(conv.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Projector, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)

        self.proj = nn.Conv2d(in_channels,
                              out_channels, 
                              kernel_size, 
                              stride, 
                              padding,
                              dilation, 
                              groups, 
                              bias)
        
        
    def forward(self, x):
        out = self.proj(x)
        return out
