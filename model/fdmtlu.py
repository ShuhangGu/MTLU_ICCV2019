from model import common
import torch
import torch.nn as nn
#from model import MTLU_CPU
from .MTLU_Package.MTLU import MTLU

def make_model(args, parent=False):
    return FDMTLU(args)


cuda_device = torch.device("cuda")

class Basic_Block(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,bias=True, bn=False):
            super(Basic_Block, self).__init__()
            m = []
            m.append(conv(n_feat,n_feat,kernel_size,bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            self.body = nn.Sequential(*m)

        
    def forward(self, x):
        	return self.body(x)


class FDMTLU(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FDMTLU, self).__init__()

        self.n_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale

        bin_number = args.bin_num
        bin_width  = args.bin_width

        bn  = args.bn




        m_head = [conv(args.n_colors*16, n_feats, kernel_size)]


        self.add_module('MTLU'+str(1),MTLU(bin_number,bin_width,n_feats))
        for i in range(self.n_blocks-2):
            self.add_module('basic_block' + str(i+2), Basic_Block(conv, n_feats, kernel_size, bn=bn))
            self.add_module('MTLU'+str(i+2),MTLU(bin_number,bin_width,n_feats))	

        self.add_module('basic_block' + str(self.n_blocks), Basic_Block(conv, n_feats, kernel_size, bn=bn))	

        m_tail = [common.SimpleGrayUpsampler(conv, 4, n_feats, act=False)]        

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #print(x)
        x = x
        residual = common.DownSamplingShuffle(x,4)
        residual  = self.head(residual)
        #residual = self.MTLU1(residual)
        residual = self.__getattr__('MTLU' + str(1))(residual)
        #print(residual.type)
        for i in range(self.n_blocks-2):
            residual = self.__getattr__('basic_block' + str(i+2))(residual)
            residual = self.__getattr__('MTLU' + str(i+2))(residual)
        residual = self.__getattr__('basic_block' + str(self.n_blocks))(residual)
        residual = self.tail(residual)	
        x = residual+x

        #x = self.add_mean(x)
        return x 



