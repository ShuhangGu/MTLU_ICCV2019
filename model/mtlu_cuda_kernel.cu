#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "caffe_cuda_macro.h"
#include "mtlu_cuda_kernel.cuh"
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))



#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>






template <typename scalar_t>
__global__ void mtlu_forward_kernel(const int nthreads, const scalar_t* const bottom_data, const scalar_t* const A_data, const scalar_t* const B_data, scalar_t*  index_mat, const scalar_t* paras,  const int feat_num, const float feat_size, scalar_t*  top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {

                    	int halfbinnum = paras[0]/2;

                        int channel_index = floor(index/feat_size);
                        channel_index     = channel_index%feat_num;
      			
                        int bin = floor(bottom_data[index]/paras[1]);
                        bin          = MIN(bin, halfbinnum-1);
                        bin          = MAX(bin,-halfbinnum);
                        int offset   = bin + halfbinnum + channel_index * paras[0];
                        top_data[index] = A_data[offset]*bottom_data[index]+B_data[offset];
			index_mat[index] = offset;

                }
	}


template <typename scalar_t>
__global__ void mtlu_backward_kernel(const int nthreads, const scalar_t* const bottom_data, 
                 const scalar_t* const top_diff, const scalar_t* const weight, const scalar_t* const index_mat, const scalar_t* paras, const int feat_num, const float feat_size, scalar_t*  bottom_diff, scalar_t*  weight_diff, scalar_t*  bias_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {


                        int offset   = floor(index_mat[index]+0.05);


                        bottom_diff[index] = weight[offset]*top_diff[index];              
                        scalar_t* temp_ptrA = &weight_diff[offset];
                        scalar_t* temp_ptrB = &bias_diff[offset];
                        atomicAdd(temp_ptrA,top_diff[index]*bottom_data[index]);

//printf("before add  f=%f\n",  temp_ptrB);
 
                        atomicAdd(temp_ptrB,top_diff[index]);
//printf("add number  f=%f\n",  top_diff[index]);
//printf("index number  f=%d\n",  index);
 
//printf("after  add  f=%f\n",  temp_ptrB);
                        
		}
	}

	// mblu_paras[0] feat_num  
	// mblu_paras[1] bin_width 
	// mblu_paras[2] bin_num
	// mblu_paras[3] count 
	// mblu_paras[4] feat_size

/*
template <typename scalar_t>
__global__ void test_kernel(scalar_t* marray)
{


}
*/


	
std::vector<at::Tensor>  mtlu_forward_cuda_call(
	at::Tensor  bottom_data, 
	at::Tensor  weight, 
	at::Tensor  bias,  
	at::Tensor  mtlu_paras,
        cudaStream_t stream) 
{


auto top_data = at::zeros_like(bottom_data);
auto index_data = at::zeros_like(bottom_data);



  int num_ = bottom_data.size(0);
  int channels_ = bottom_data.size(1);
  int height_ = bottom_data.size(2);
  int width_ = bottom_data.size(3);
  float feat_size = height_*width_;
  int dim = num_*channels_*height_*width_;


	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.type(), "mtlu_forward_cuda", 
			([&] {

        			mtlu_forward_kernel<scalar_t><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0 ,  stream>>>(
				dim,
            			bottom_data.data<scalar_t>(),
            			weight.data<scalar_t>(),
            			bias.data<scalar_t>(),
            			index_data.data<scalar_t>(),
            			mtlu_paras.data<scalar_t>(),
            			channels_,
            			feat_size,
            			top_data.data<scalar_t>()
          			);
   			      }
			)
			);
 
  return {top_data,index_data};
}



std::vector<at::Tensor>   mtlu_backward_cuda_call(
	at::Tensor   bottom_data, 
	at::Tensor   weight, 
	at::Tensor   top_diff, 
	at::Tensor  index_mat, 
	at::Tensor   mtlu_paras, 
	cudaStream_t stream) 
{




auto bottom_diff = at::zeros_like(bottom_data);
auto weight_diff = at::zeros_like(weight);
auto bias_diff = at::zeros_like(weight);



  int num_ = bottom_data.size(0);
  int channels_ = bottom_data.size(1);
  int height_ = bottom_data.size(2);
  int width_ = bottom_data.size(3);
  float feat_size = height_*width_;
  int dim = num_*channels_*height_*width_;




	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.type(), "lltm_forward_cuda", 
			([&] {

        			mtlu_backward_kernel<scalar_t><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0 ,  stream>>>(
				dim,
            			bottom_data.data<scalar_t>(),
            			top_diff.data<scalar_t>(),
            			weight.data<scalar_t>(),
            			index_mat.data<scalar_t>(),
            			mtlu_paras.data<scalar_t>(),
            			channels_,
            			feat_size,
            			bottom_diff.data<scalar_t>(),
            			weight_diff.data<scalar_t>(),
            			bias_diff.data<scalar_t>()
          			);
   			      }
			)
			);
 
return {bottom_diff, weight_diff, bias_diff};
}
	
	

