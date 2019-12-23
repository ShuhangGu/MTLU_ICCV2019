import torch
import torch.nn as nn
from torch.autograd import Variable
import mtlu_cuda
import timeit

class MTLU_AF(torch.autograd.Function):
	@staticmethod
	def forward(self, x, weight, bias, paras):


		outputs = mtlu_cuda.forward(x, weight, bias, paras)
		y = outputs[0]
		indexmat = outputs[1]
		#print(indexmat)
		self.save_for_backward(x, weight, bias, paras, indexmat)
		return y

	@staticmethod
	def backward(self, grad_output):
		#starter = timeit.default_timer()
		x, weight, bias, paras, indexmat = self.saved_tensors
		grad_paras = None
		outputs =  mtlu_cuda.backward(x, weight, grad_output.data, indexmat, paras)			
		grad_input, grad_weight, grad_bias = outputs
		return grad_input, grad_weight, grad_bias, grad_paras

	# mblu_paras[0] feat_num  
	# mblu_paras[1] bin_width 
	# mblu_paras[2] bin_num
	# mblu_paras[3] count 
	# mblu_paras[4] feat_size

class MTLU(nn.Module):
	def __init__(self,BinNum=40,BinWidth=0.05, FeatMapNum=64):
		super(MTLU, self).__init__()
		self.weight = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
		self.bias   = nn.Parameter(torch.zeros(FeatMapNum, BinNum))
		HalfBinNum = int(BinNum/2)
		self.weight.data[:,HalfBinNum:]=1
		self.MTLUpara = nn.Parameter(torch.zeros(2))
		self.MTLUpara.data[0] = BinNum
		self.MTLUpara.data[1] = BinWidth
	
	
	def forward(self, x):
		return MTLU_AF.apply(x, self.weight, self.bias, self.MTLUpara)



