#include "../headers/export.h"
#include "../headers/hofs.h"
#include "../funcptrs/builtin_tabulate_and_map_int.h"
#include "../funcptrs/user_tabulate_int.h"
#include "../funcptrs/builtin_map_int.h"
#include "../funcptrs/user_map_int.h"
#include "../funcptrs/builtin_reduce_and_scan_int.h"
#include "../funcptrs/user_reduce_int.h"
#include <stdio.h>
#include <time.h>

#define blockSize = 256

#define threads_reduce 1024
#define block_red_size_reduce (threads_reduce / 32)

#define threads_scan 1024
#define block_red_size_scan (threads_scan / 32)

//Tabulate
__global__ 
void tabulate_int_kernel(int* arr, int len, tabulate_fun_int f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr[idx] = f(idx);

}
extern "C"
void* tabulate_int(void* f, int size){
  
  tabulate_fun_int hof = (tabulate_fun_int)f;
  
  void* dev_ptr;
  cudaMalloc(&dev_ptr, sizeof(int) * size);

  int blockNum = (size / 256) + 1;
  tabulate_int_kernel<<<blockNum, 256>>>((int*)dev_ptr, size, hof);
  
  return dev_ptr;

}

//Map
__global__
void map_int_kernel(int* arr, int len, map_fun_int f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr[idx] = f(arr[idx]);

}
extern "C"
void* map_int(void* inarr, void* f, int size){
  
  map_fun_int hof = (map_fun_int)f;      
  int blockNum = (size / 256) + 1;
  
  map_int_kernel<<<blockNum, 256>>>((int*)inarr, size, hof);

  return inarr;

}


__inline__ __device__
int warp_red_int(int t, reduce_fun_int f){
  int res = t;

  #pragma unroll
  for(int i = warpSize / 2;i > 0;i /= 2){
    int a = __shfl_down(res, i);
    res = f(res, a);
    //res += a;
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

  int inter_res = warp_red_int(t, f);
  
  if(localIdx == 0){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();
  
  int broadval2 = (threadIdx.x < block_red_size_reduce) ? warp_reds[localIdx] : b;
  int res = b;
  if(warpIdx == 0){
    res = warp_red_int(broadval2, f);
  }

  return res;
}

__global__
void reduce_int_kernel(int* in, int* out, int size, int b, reduce_fun_int f){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = b;
  
  #pragma unroll
  for(int i = idx;i < size;i += blockDim.x * gridDim.x){
    sum = f(sum,in[i]);
    //sum += in[i];
  }
  
  sum = reduce_block_int(sum, b, f);
  
  if(threadIdx.x == 0){
    out[blockIdx.x] = sum;
  }
  
}

// cite : https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler
// for algorithm / ideas on how to use shfl methods for fast reductions
extern "C"
int reduce_int_shfl(void* arr, int size, int b, void* f){

  reduce_fun_int hof = (reduce_fun_int)f;
  

  int numBlocks = (size / threads_reduce) + 1;
  void* res;
  cudaMalloc(&res, sizeof(int) * numBlocks);
  reduce_int_kernel<<<numBlocks, threads_reduce>>>((int*)arr, (int*)res, 
                                                   size, b, hof);
  reduce_int_kernel<<<1, 1024>>>((int*)res, (int*)res, numBlocks, b, hof);

  int ret;
  cudaMemcpy(&ret, res, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(res);
  return ret;
}

//BEGIN SCAN

__device__ __inline__
int warp_scan_shfl(int b, scan_fun_int f, int* out, int idx, int length){
  int warpIdx = threadIdx.x % warpSize;
  int res;
  if(idx < length){
    res = out[idx];
  }
  else{
    res = b;
  }
  #pragma unroll
  for(int i = 1;i < warpSize;i *= 2){
    int a = __shfl_up(res, i);
    if(i <= warpIdx){
      res = f(a, res);
    }
  }
  if(idx < length){
    out[idx] = res;
  }
  return res;
}

__device__ __inline__
int block_scan(int* in, int length, scan_fun_int f, int b){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int warp_reds[block_red_size_scan];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx= threadIdx.x % warpSize;

  int inter_res = warp_scan_shfl(b, f, in, idx, length);

  if(localIdx == warpSize - 1){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();

  int res = b;
  if(warpIdx == 0){
    res = warp_scan_shfl(b, f, warp_reds, localIdx, block_red_size_scan);
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
void scan_int_kernel(int* in, int* block_results, scan_fun_int f, int b, int length){
  
  int block_res = block_scan(in, length, f, b);
  if(threadIdx.x == block_red_size_scan - 1){
    block_results[blockIdx.x] = block_res;
  }
}
__global__
void compress_results(int* block_res, int* out, int len, scan_fun_int f){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(blockIdx.x == 0){
    return;
  }
  else{
    if(idx < len){
      out[idx] = f(block_res[blockIdx.x - 1], out[idx]);
    }
  }
}

//this is terrible
__global__
void serial_scan(int* bres, int len, int b, scan_fun_int f){
  int res = b;
  #pragma unroll
  for(int i = 0;i < len;i++){
    res = f(res, bres[i]);
    bres[i] = res;
  }
}

extern "C"
void* inclusive_scan_int(void* in, void* f, int length, int b){
  
  scan_fun_int hof = (scan_fun_int)f;

  int num_blocks_first = (length / threads_scan) + 1;
  int* block_results;
  int* dummy;
  cudaMalloc(&block_results, sizeof(int) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(int));

  scan_int_kernel<<<num_blocks_first, threads_scan>>>
          ((int*)in, block_results, hof, b, length);

  if(num_blocks_first == 1){
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    return in;
  }
  else if(num_blocks_first <= 1024){
    scan_int_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
    compress_results<<<num_blocks_first, threads_scan>>>(block_results, (int*)in, length, hof);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    return in;
  }
  else{
    int leftover = (num_blocks_first / threads_scan) + 1;
    int* block_block_results;
    cudaMalloc(&block_block_results, sizeof(int) * leftover);
    scan_int_kernel<<<leftover, threads_scan>>>
            (block_results, block_block_results, hof, b, num_blocks_first);
    serial_scan<<<1,1>>>(block_block_results, leftover, b, hof);
    compress_results<<<leftover, threads_scan>>>
            (block_block_results, block_results, num_blocks_first, hof);
    compress_results<<<num_blocks_first, threads_scan>>>(block_results, (int*)in, length, hof);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(block_block_results);
    return in;
  }
}

//BEGIN EXCLUSIVE SCAN

__device__ __inline__
int warp_scan_shfl_exl(int b, scan_fun_int f, int* out, int idx, int length){
  int warpIdx = threadIdx.x % warpSize;
  int res;
  if(warpIdx == 0){
    res = b;
  }
  else{
    if(idx - 1 > length){
      res = b;
    }
    else{
      res = out[idx - 1];
    }
  }
  #pragma unroll
  for(int i = 1;i < warpSize;i *= 2){
    int a = __shfl_up(res, i);
    if(i <= warpIdx){
      res = f(a, res);
    }
  }
  if(idx < length){
    out[idx] = res;
  }
  return res;
}

__device__ __inline__
int excl_block_scan(int* in, int length, scan_fun_int f, int b){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int warp_reds[block_red_size_scan];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx= threadIdx.x % warpSize;

  int inter_res = warp_scan_shfl(b, f, in, idx, length);

  if(localIdx == warpSize - 1){
    warp_reds[warpIdx] = inter_res;
  }

  __syncthreads();

  int res = b;
  if(warpIdx == 0){
    res = warp_scan_shfl(b, f, warp_reds, localIdx, block_red_size_scan);
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
void excl_scan_int_kernel(int* in, int* block_results, scan_fun_int f, int b, int length, int* out){
  
  int block_res = block_scan(in, length, f, b);
  if(threadIdx.x == block_red_size_scan - 1){
    block_results[blockIdx.x] = block_res;
    *out = block_res;
  }
}
__global__
void excl_compress_results(int* block_res, int* out, int len, scan_fun_int f, int* final, int b){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= len) return;
  if(blockIdx.x != 0){
    out[idx] = f(block_res[blockIdx.x - 1], out[idx]);
  }
  __syncthreads();
  int toWrite = b;
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
int exclusive_scan_int(void* in, void* f, int length, int b){
  
  scan_fun_int hof = (scan_fun_int)f;

  int num_blocks_first = (length / threads_scan) + 1;
  int* block_results;
  int* dummy;
  int* final_val;
  cudaMalloc(&block_results, sizeof(int) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(int));
  cudaMalloc(&final_val, sizeof(int));


  scan_int_kernel<<<num_blocks_first, threads_scan>>>
          ((int*)in, block_results, hof, b, length);
  int res;
  if(num_blocks_first == 1){
    excl_compress_results<<<num_blocks_first, threads_scan>>>
          (block_results, (int*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    return res;
  }
  else if(num_blocks_first <= 1024){
    scan_int_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
    excl_compress_results<<<num_blocks_first, threads_scan>>>
            (block_results, (int*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    return res;
  }
  else{
    int leftover = (num_blocks_first / threads_scan) + 1;
    int* block_block_results;
    cudaMalloc(&block_block_results, sizeof(int) * leftover);
    scan_int_kernel<<<leftover, threads_scan>>>
            (block_results, block_block_results, hof, b, num_blocks_first);
    serial_scan<<<1,1>>>(block_block_results, leftover, b, hof);
    compress_results<<<leftover, threads_scan>>>
            (block_block_results, block_results, num_blocks_first, hof);
    excl_compress_results<<<num_blocks_first, threads_scan>>>
            (block_results, (int*)in, length, hof, final_val, b);
    cudaMemcpy(&res, final_val, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    cudaFree(final_val);
    cudaFree(block_block_results);
    return res;
  }
}

/*
//Reduce - cite http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__global__
void reduce_int_kernel(int* arr, int len, reduce_fun_int f){
  
  extern __shared__ int sdata[];
  unsigned int blockSize = blockDim.x;
  unsigned int thread_id = threadIdx.x;
  unsigned int array_id = blockIdx.x * (blockDim.x * 2) + thread_id;
  unsigned int gridSize = 2 * blockDim.x * gridDim.x;
  sdata[thread_id] = 0;

  while (array_id < len) 
  {
    sdata[thread_id] += arr[i] + arr[array_id+ blockSize];
    array_id += blockSize;
  }

  __syncthreads();

  if(thread_id < 128) 
    sdata[thread_id] += sdata[thread_id + 128];

  __syncthreads();

  if (thread_id <  64)
    sdata[thread_id] += sdata[thread_id + 64];

  __syncthreads();

  if (thread_id < 32)
  {
    sdata[thread_id] += sdata[thread_id + 32];
    sdata[thread_id] += sdata[thread_id + 16];
    sdata[thread_id] += sdata[thread_id + 8];
    sdata[thread_id] += sdata[thread_id + 4];
    sdata[thread_id] += sdata[thread_id + 1];
  }
  
  if (thread_id ==0)
    arr[blockIdx.x] = sdata[0];

}

void* reduce_int(void* inarr, void* f, int size){
  
  reduce_fun_int hof = (reduce_fun_int)f;      
  int blockNum = (size / 256) + 1;
  
  reduce_int_kernel<<<blockNum, 256>>>((int*)inarr, size, hof);

  return inarr;

}
*/
