#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <iostream>

#include "mtlu_cuda_kernel.cuh"

std::vector<at::Tensor>  mtlu_forward_cuda(at::Tensor  bottom_data_tensor, at::Tensor weight_tensor, at::Tensor bias_tensor,  at::Tensor mtlu_paras_tensor)
{




  return mtlu_forward_cuda_call(
	bottom_data_tensor, 
	weight_tensor, 
	bias_tensor, 
	mtlu_paras_tensor,
    	at::cuda::getCurrentCUDAStream());
 



}

std::vector<at::Tensor>   mtlu_backward_cuda(at::Tensor  bottom_data_tensor, at::Tensor  weight_tensor, at::Tensor  top_diff_tensor, at::Tensor  index_tensor,
at::Tensor  mtlu_paras_tensor)
{




  return mtlu_backward_cuda_call(
	bottom_data_tensor, 
	weight_tensor, 
	top_diff_tensor, 
	index_tensor,
	mtlu_paras_tensor, 
        at::cuda::getCurrentCUDAStream()
	);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mtlu_forward_cuda, "MTLU forward (CUDA)");
  m.def("backward", &mtlu_backward_cuda, "MTLU backward (CUDA)");
}

