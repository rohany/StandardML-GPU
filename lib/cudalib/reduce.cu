#include "../headers/hofs.h"
#include "../headers/export.h"

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

__global__
void reduce_block_int(int* in, int* out, int size, int b, reduce_fun_int f){
  
  // assuming warp size is 32
  // can fix later in the kernel call
  __shared__ int warp_reds[block_red_size_reduce];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx = threadIdx.x % warpSize;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // need to handle case where length of array is not
  // exactly equal to the block size

  int broadval = (idx < size) ? in[idx] : b;
  int inter_res = warp_red_int(broadval, f);
  
  if(localIdx == 0){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();
  
  int broadval2 = (localIdx < block_red_size_reduce) ? warp_reds[localIdx] : b;
  if(warpIdx == 0){
    int res = warp_red_int(broadval2, f);
    if(idx == 0){
      *out = res;
    }
  }
}

extern "C"
int reduce_int(void* arr, int size, int b, void* f){
  //reduce_fun_int hof = (reduce_fun_int)f;
  
  return 0;
}
