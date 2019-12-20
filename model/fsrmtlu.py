from model import common
import torch
import torch.nn as nn
#from model import MTLU_CPU
from .MTLU_Package.MTLU import MTLU

def make_model(args, parent=False):
    return FSRMTLU(args)




class Basic_Block(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,bias=True, bn=False):
            super(Basic_Block, self).__init__()
            m = []
            m.append(conv(n_feat,n_feat,kernel_size,bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            self.body = nn.Sequential(*m)
            # No activation function here!!!	
        
    def forward(self, x):
        	return self.body(x)


class FSRMTLU(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FSRMTLU, self).__init__()

        self.m_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3 

        bin_number = args.bin_num
        bin_width  = args.bin_width
        scale = args.scale
        #act = nn.PReLU(n_feats)
        #act = MTLU_GPU(40,0.05,64)
        bn  = args.bn
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        

        m_head = [conv(args.n_colors, n_feats, kernel_size)]



        self.add_module('MTLU'+str(1),MTLU(bin_number,bin_width,n_feats))
        for i in range(self.m_blocks-2):
            self.add_module('basic_block' + str(i+2), Basic_Block(conv, n_feats, kernel_size, bn=bn))
            self.add_module('MTLU'+str(i+2),MTLU(bin_number,bin_width,n_feats))	

        self.add_module('basic_block' + str(self.m_blocks), Basic_Block(conv, n_feats, kernel_size, bn=bn))	


        m_tail = [common.SimpleUpsampler(conv, scale, n_feats, act=False)]        
        self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)
        x = self.__getattr__('MTLU' + str(1))(x)
        res = x.clone()
        for i in range(self.m_blocks-2):
            res = self.__getattr__('basic_block' + str(i+2))(res)
            res = self.__getattr__('MTLU' + str(i+2))(res)
        res = self.__getattr__('basic_block' + str(self.m_blocks))(res)
	
        res += x
        x = self.tail(res)
        #x = self.add_mean(x)
        return x 


