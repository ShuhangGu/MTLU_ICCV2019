#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>
#include "ATen/cuda/CUDAContext.h"


std::vector<at::Tensor>  mtlu_forward_cuda_call(
	at::Tensor  bottom_data, 
	at::Tensor  weight, 
	at::Tensor  bias,  
	at::Tensor  mtlu_paras,
        cudaStream_t stream) ; 


std::vector<at::Tensor>   mtlu_backward_cuda_call(
	at::Tensor   bottom_data, 
	at::Tensor   weight, 
	at::Tensor   top_diff, 
	at::Tensor  index_mat, 
	at::Tensor   mtlu_paras, 
	cudaStream_t stream) ; 

