#include "../headers/export.h"
#include "../headers/hofs.h"

//Builtin function pointers
#include "../funcptrs/builtin_tabulate_and_map_float.h"
#include "../funcptrs/builtin_reduce_and_scan_float.h"
#include "../funcptrs/builtin_filter_float.h"

//User defined function pointers
#include "../funcptrs/user_tabulate_float.h"
#include "../funcptrs/user_map_float.h"
#include "../funcptrs/user_reduce_float.h"
#include "../funcptrs/user_scan_float.h"
#include "../funcptrs/user_filter_float.h"
#include "../funcptrs/user_zipwith_float.h"

#include <stdio.h>
#include <time.h>

#define blockSize = 256

#define threads_reduce 1024
#define block_red_size_reduce (threads_reduce / 32)

#define threads_scan 1024
#define block_red_size_scan (threads_scan / 32)

#define threads_filter 256

//Tabulate
__global__ 
void tabulate_float_kernel(float* arr, int len, tabulate_fun_float f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr[idx] = f(idx);
}

extern "C"
void* tabulate_float(int size, void* f){
  
  tabulate_fun_float hof = (tabulate_fun_float)f;
  
  void* dev_ptr;
  cudaMalloc(&dev_ptr, sizeof(float) * size);

  int blockNum = (size / 256) + 1;
  tabulate_float_kernel<<<blockNum, 256>>>((float*)dev_ptr, size, hof);
  cudaDeviceSynchronize();
  return dev_ptr;

}

//Map
__global__
void map_float_kernel(float* arr, int len, map_fun_float f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr[idx] = f(arr[idx]);

}

extern "C"
void* map_float(void* inarr, void* f, int size){
  
  map_fun_float hof = (map_fun_float)f;      
  int blockNum = (size / 256) + 1;
  
  map_float_kernel<<<blockNum, 256>>>((float*)inarr, size, hof);
  cudaDeviceSynchronize();

  return inarr;

}


__inline__ __device__
float warp_red_float(float t, reduce_fun_float f){
  float res = t;
  
  float a = __shfl_down(res, 16);
  res = f(res, a);

  a = __shfl_down(res, 8);
  res = f(res, a);
  
  a = __shfl_down(res, 4);
  res = f(res, a);
  
  a = __shfl_down(res, 2);
  res = f(res, a);
  
  a = __shfl_down(res, 1);
  res = f(res, a);
  
  return res;
}

__inline__ __device__
float reduce_block_float(float t, float b, reduce_fun_float f){
  
  // assuming warp size is 32
  // can fix later in the kernel call
  __shared__ float warp_reds[block_red_size_reduce];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx = threadIdx.x % warpSize;

  float inter_res = warp_red_float(t, f);
  
  if(localIdx == 0){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();
  
  float broadval2 = (threadIdx.x < block_red_size_reduce) ? warp_reds[localIdx] : b;
  float res = b;
  if(warpIdx == 0){
    res = warp_red_float(broadval2, f);
  }

  return res;
}

__global__
void reduce_float_kernel(float* in, float* out, int size, float b, reduce_fun_float f){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = b;
  
  #pragma unroll
  for(int i = idx;i < size;i += blockDim.x * gridDim.x){
    sum = f(sum,in[i]);
  }
  
  sum = reduce_block_float(sum, b, f);
  
  if(threadIdx.x == 0){
    out[blockIdx.x] = sum;
  }
  
}

// cite : https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler
// for algorithm / ideas on how to use shfl methods for fast reductions
extern "C"
float reduce_float_shfl(void* arr, int size, float b, void* f){

  reduce_fun_float hof = (reduce_fun_float)f;
  

  int numBlocks = (size / threads_reduce) + 1;
  void* res;
  cudaMalloc(&res, sizeof(float) * numBlocks);
  reduce_float_kernel<<<numBlocks, threads_reduce>>>((float*)arr, (float*)res, 
                                                   size, b, hof);
  reduce_float_kernel<<<1, 1024>>>((float*)res, (float*)res, numBlocks, b, hof);

  float ret;
  cudaMemcpy(&ret, res, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(res);
  return ret;
}

//BEGIN SCAN

__device__ __inline__
float warp_scan_shfl_float(float b, scan_fun_float f, float* out, int idx, int length){
  int warpIdx = threadIdx.x % warpSize;
  float res;
  if(idx < length){
    res = out[idx];
  }
  else{
    res = b;
  }

  float a = __shfl_up(res, 1);
  if(1 <= warpIdx){
    res = f(a, res);
  }

  a = __shfl_up(res, 2);
  if(2 <= warpIdx){
    res = f(a, res);
  }

  a = __shfl_up(res, 4);
  if(4 <= warpIdx){
    res = f(a, res);
  }

  a = __shfl_up(res, 8);
  if(8 <= warpIdx){
    res = f(a, res);
  }

  a = __shfl_up(res, 16);
  if(16 <= warpIdx){
    res = f(a, res);
  }

  if(idx < length){
    out[idx] = res;
  }
  return res;
}

__device__ __inline__
float block_scan_float(float* in, int length, scan_fun_float f, float b){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float warp_reds[block_red_size_scan];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx= threadIdx.x % warpSize;

  float inter_res = warp_scan_shfl_float(b, f, in, idx, length);

  if(localIdx == warpSize - 1){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();

  float res = b;
  if(warpIdx == 0){
    res = warp_scan_shfl_float(b, f, warp_reds, localIdx, block_red_size_scan);
  }
  
  __syncthreads();

  if(idx < length && warpIdx != 0){
    in[idx] = f(warp_reds[warpIdx - 1], in[idx]);
  }

  //warp number 0, lane number block_red_size_scan 
  //will return the final result for scanning over this
  //block 
  return res;
}

//inclusive kernel
__global__
void scan_float_kernel(float* in, float* block_results, scan_fun_float f, float b, int length){
  
  float block_res = block_scan_float(in, length, f, b);
  if(threadIdx.x == block_red_size_scan - 1){
    block_results[blockIdx.x] = block_res;
  }
}
__global__
void compress_results_float(float* block_res, float* out, int len, scan_fun_float f){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(blockIdx.x == 0){
    return;
  }
  else{
    if(idx < len){
      // TODO : load this into shared memory or something?
      out[idx] = f(block_res[blockIdx.x - 1], out[idx]);
    }
  }
}

//this is terrible
__global__
void serial_scan_float(float* bres, int len, float b, scan_fun_float f){
  float res = b;
  #pragma unroll
  for(int i = 0;i < len;i++){
    res = f(res, bres[i]);
    bres[i] = res;
  }
}

extern "C"
void* inclusive_scan_float(void* in, void* f, int length, float b){
  
  scan_fun_float hof = (scan_fun_float)f;

  int num_blocks_first = (length / threads_scan) + 1;
  float* block_results;
  float* dummy;
  cudaMalloc(&block_results, sizeof(float) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(float));

  scan_float_kernel<<<num_blocks_first, threads_scan>>>
          ((float*)in, block_results, hof, b, length);

  if(num_blocks_first == 1){
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    return in;
  }
  else if(num_blocks_first <= 1024){
    scan_float_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
    compress_results_float<<<num_blocks_first, threads_scan>>>
            (block_results, (float*)in, length, hof);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    return in;
  }
  else{
    int leftover = (num_blocks_first / threads_scan) + 1;
    float* block_block_results;
    cudaMalloc(&block_block_results, sizeof(float) * leftover);
    scan_float_kernel<<<leftover, threads_scan>>>
            (block_results, block_block_results, hof, b, num_blocks_first);
    serial_scan_float<<<1,1>>>(block_block_results, leftover, b, hof);
    compress_results_float<<<leftover, threads_scan>>>
            (block_block_results, block_results, num_blocks_first, hof);
    compress_results_float<<<num_blocks_first, threads_scan>>>
            (block_results, (float*)in, length, hof);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(block_block_results);
    return in;
  }
}

//BEGIN EXCLUSIVE SCAN

__global__
void excl_compress_results_float(float* block_res, float* out, int len, 
                                 scan_fun_float f, float* final, float b){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= len) return;
  if(blockIdx.x != 0){
    out[idx] = f(block_res[blockIdx.x - 1], out[idx]);
  }
  __syncthreads();
  float toWrite = b;
  if(threadIdx.x == 0){
    if(idx == 0){
      toWrite = b;
    }
    else{
      toWrite = block_res[blockIdx.x - 1];
    }
  }
  else{
    toWrite = out[idx - 1];
  }
  if(idx == len - 1){
    *final = out[idx];
  }
  __syncthreads();
  out[idx] = toWrite;
}

extern "C"
float exclusive_scan_float(void* in, void* f, int length, float b){
  
  scan_fun_float hof = (scan_fun_float)f;

  int num_blocks_first = (length / threads_scan) + 1;
  float* block_results;
  float* dummy;
  float* final_val;
  cudaMalloc(&block_results, sizeof(float) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(float));
  cudaMalloc(&final_val, sizeof(float));


  scan_float_kernel<<<num_blocks_first, threads_scan>>>
          ((float*)in, block_results, hof, b, length);
  float res;
  if(num_blocks_first == 1){
    excl_compress_results_float<<<num_blocks_first, threads_scan>>>
          (block_results, (float*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    return res;
  }
  else if(num_blocks_first <= 1024){
    scan_float_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
    excl_compress_results_float<<<num_blocks_first, threads_scan>>>
            (block_results, (float*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    return res;
  }
  else{
    int leftover = (num_blocks_first / threads_scan) + 1;
    float* block_block_results;
    cudaMalloc(&block_block_results, sizeof(float) * leftover);
    scan_float_kernel<<<leftover, threads_scan>>>
            (block_results, block_block_results, hof, b, num_blocks_first);
    serial_scan_float<<<1,1>>>(block_block_results, leftover, b, hof);
    compress_results_float<<<leftover, threads_scan>>>
            (block_block_results, block_results, num_blocks_first, hof);
    excl_compress_results_float<<<num_blocks_first, threads_scan>>>
            (block_results, (float*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    cudaFree(block_block_results);
    return res;
  }
}

__global__
void filter_map_float(float* in, float* out1, int len, filter_fun_float f){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < len){
    if(f(in[idx])){
      out1[idx] = 1;
    }
    else{
      out1[idx] = 0;
    }
  }
}
__global__
void squish_float(float* in, float* scanned, float* out, int length, filter_fun_float f){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(idx < length){
    if(f(in[idx]) == 1){
      // floating point arith please
      out[(int)scanned[idx]] = in[idx];
    }
  }
}

extern "C"
void* filter_float(void* arr, int length, void* f, Pointer outlen){
  filter_fun_float hof = (filter_fun_float)f;
  
  int blocks = (length / threads_filter) + 1;
    
  // make buffer array

  // this map could have been fused in with the scan with some 
  // extra code copy pasta i didnt want to do

  float* scanned;
  cudaMalloc(&scanned, sizeof(float) * length);
  filter_map_float<<<blocks, threads_filter>>>((float*)arr, scanned, length, hof);
  
  //scan over the bits
  reduce_fun_float add = (reduce_fun_float)gen_add_float();
  // - need the integer scan ..., but dont have rdc on...
  int len = (int)exclusive_scan_float(scanned, (void*)add, length, 0);

  float* res;
  cudaMalloc(&res, sizeof(float) * len);

  squish_float<<<blocks, threads_filter>>>((float*)arr, scanned, res, length, hof);
  *(int*)outlen = len;

  cudaFree(scanned);
  return res;
}

__global__
void zipsquish_float(float* arr1, float* arr2, float* out, zipwith_fun_float f, int length){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < length){
    out[idx] = f(arr1[idx], arr2[idx]);
  }
}

extern "C"
void* zipwith_float(void* arr1, void* arr2, void* f, int length){

  zipwith_fun_float hof = (zipwith_fun_float)f;
  
  float* res;
  cudaMalloc(&res, sizeof(int) * length);

  int blocks = (length / threads_filter) + 1;
  zipsquish_float<<<blocks, threads_filter>>>((float*)arr1, (float*)arr2, res, hof, length);

  cudaDeviceSynchronize();
  return res;
}
//Reduce - cite http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf - another reduction algorithm choice
