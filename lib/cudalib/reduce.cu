#include "../headers/hofs.h"
#include "../headers/export.h"
#include <stdio.h>

#define threads_reduce 256
#define block_red_size_reduce (threads_reduce / 32)

__inline__ __device__
int warp_red_int(int t, reduce_fun_int f){
  int res = t;
  for(int i = warpSize / 2;i >= 1;i /= 2){
    int a = __shfl_down(res, i);
    res = (*f)(res, a);
  }
  return res;
}

__inline__ __device__
int reduce_block_int(int t, int b, reduce_fun_int f){
  
  // assuming warp size is 32
  // can fix later in the kernel call
  __shared__ int warp_reds[block_red_size_reduce];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx = threadIdx.x % warpSize;

  // need to handle case where length of array is not
  // exactly equal to the block size

  int inter_res = warp_red_int(t, f);
  
  if(localIdx == 0){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();
  
  int broadval2 = (localIdx < block_red_size_reduce) ? warp_reds[localIdx] : b;
  int res = b;
  if(warpIdx == 0){
    res = warp_red_int(broadval2, f);
  }

  return res;
}

__global__
void reduce_int_kernel(int* in, int* out, int size, int b, reduce_fun_int f){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx == 0){
    printf("got here\n");
  }
  int sum = b;

  for(int i = idx;i < size;i += blockDim.x * gridDim.x){
    sum = (*f)(b,sum);
  }
  
  //sum = reduce_block_int(sum, b, f);
  /*
  if(threadIdx.x == 0){
    while(true){
      int t = *out;
      int final = (*f)(sum, t);
      int res = atomicCAS(out, t, final);
      if(res != t){
        break;
      }
    }
  }
  */
}


extern "C"
int reduce_int_shfl(void* arr, int size, int b, void* f){
  reduce_fun_int hof = (reduce_fun_int)f;
  
  void* res;
  cudaMalloc(&res, sizeof(int));

  int numBlocks = (size / threads_reduce) + 1;
  reduce_int_kernel<<<numBlocks, threads_reduce>>>((int*)arr, (int*)res, 
                                                   size, b, hof);

  cudaDeviceSynchronize();
  cudaError_t wei = cudaGetLastError();
  printf("%s\n", cudaGetErrorString(wei));

  int ret;
  cudaMemcpy(&ret, res, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", ret);
  return ret;
}
