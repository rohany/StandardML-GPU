#include "../headers/export.h"
#include "../headers/hofs.h"
#include "../funcptrs/builtin_tabulate_and_map_intxint.h"
#include "../funcptrs/user_tabulate_int_tuple.h"
#include "../funcptrs/user_map_int_tuple.h"
#include "../funcptrs/builtin_reduce_and_scan_int_tuple.h"
#include "../funcptrs/user_reduce_int_tuple.h"
#include "../funcptrs/user_scan_int_tuple.h"
#include "../funcptrs/builtin_filter_int_tuple.h"
#include "../funcptrs/user_filter_int_tuple.h"
#include "../funcptrs/user_zipwith_int_tuple.h"
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
void tabulate_int_tuple_kernel(int* arr_1, int* arr_2, int len, tabulate_fun_int_tuple f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr_1[idx] = f(idx);
  arr_2[idx] = f(idx);
}

extern "C"
void* tabulate_int_tuple(int size, void* f, Pointer dev_ptr_1, Pointer dev_ptr_1){
  
  tabulate_fun_int_tuple hof = (tabulate_fun_int_tuple)f;
  
  cudaMalloc(&(void*)dev_ptr_1, sizeof(int) * size);
  cudaMalloc(&(void*)dev_ptr_2, sizeof(int) * size);

  int blockNum = (size / 256) + 1;
  tabulate_int_tuple_kernel<<<blockNum, 256>>>((int*)dev_ptr_1, (int*)dev_ptr_2, size, hof);
  cudaDeviceSynchronize();

}

//Map
__global__
void map_int_tuple_kernel(int* arr_1, int* arr_2 int len, map_fun_int_tuple f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  std::pair<int, int> tuple = f(arr_1[idx],arr_2[idx]);
  arr_1[idx] = tuple.first
  arr_2[idx] = tuple.second
}
extern "C"
std::pair<void*, void*> map_int_tuple(void* inarr_1, void* inarr_2, void* f, int size){
  
  map_fun_int_tuple hof = (map_fun_int_tuple)f;      
  int blockNum = (size / 256) + 1;
  
  map_intxint_kernel<<<blockNum, 256>>>((int*)inarr_1, (int*)inarr_2, size, hof);

  std::pair<int, int> ret = pair(inarr_1, inarr_2)

  return std::make_pair(inarr_1, inarr_2);

}


__inline__ __device__
std::pair<int,int> warp_red_int_tuple(int t_1, int t_2, reduce_fun_int_tuple f){
  int res_1 = t_1;
  int res_2 = t_2;

  #pragma unroll
  for(int i = warpSize / 2;i > 0;i /= 2){
    int a = __shfl_down(res_1, res_2, i);
    res = f(res_1, res_2, a);
    res_1 = res.first();
    res_2 = res.second();

    //res += a;
  }

  return res;
}

__inline__ __device__
std::pair<int,int> reduce_block_int_tuple(int t_1, int t_2, int b_1, int b_2, reduce_fun_int_tuple f){
  
  // assuming warp size is 32
  // can fix later in the kernel call
  __shared__ int warp_reds_1[block_red_size_reduce];
  __shared__ int warp_reds_2[block_red_size_reduce];

  int warpIdx = threadIdx.x / warpSize;

  int localIdx = threadIdx.x % warpSize;

  std::pair<int,int> inter_res = warp_red_int_tuple(t_1, t_2, f);
  
  if(localIdx == 0){
    warp_reds_1[warpIdx] = inter_res.first;
    warp_reds_2[warpIdx] = inter_res.second;
  }

  __syncthreads();
  
  int broadval2_1 = (threadIdx.x < block_red_size_reduce) ? warp_reds_1[localIdx] : b_1;
  int broadval2_2 = (threadIdx.x < block_red_size_reduce) ? warp_reds_2[localIdx] : b_2;

  std::pair<int, int> res = std::make_pair(b_1, b_2);
  if(warpIdx == 0){
    res = warp_red_int_tuple(broadval2_1, broadval2_2, f);
  }

  return res;
}

__global__
void reduce_int_tuple_kernel(int* in_1, int* in_2, int* out_1, int* our_2, int size, int b_1, int b_2, reduce_fun_int_tuple f){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int sum_1 = b_1;
  int sum_2 = b_2;
  std::pair<int, int> sum;
  
  #pragma unroll
  for(int i = idx; i < size; i += blockDim.x * gridDim.x){
    sum = f(sum_1, sum_2, in_1[i], in_2[i]);
    sum_1 = sum.first;
    sum_2 = sum.second;
    //sum += in[i];
  }
  
  sum = reduce_block_int_tuple(sum_1, sum_2, b_1, b_2, f);
  
  if(threadIdx.x == 0){
    out_1[blockIdx.x] = sum.first;
    out_2[blockIdx.x] = sum.second;
  }
  
}

// cite : https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler
// for algorithm / ideas on how to use shfl methods for fast reductions
extern "C"
std::pair<int,int> reduce_int_tuple_shfl(void* arr_1, void* arr_2, int size, int b_1, int b_2, void* f){

  reduce_fun_int_tuple hof = (reduce_fun_int_tuple) f;
  

  int numBlocks = (size / threads_reduce) + 1;
  void* res_1;
  void* res_2;

  cudaMalloc(&res_1, sizeof(int) * numBlocks);
  cudaMalloc(&res_2, sizeof(int) * numBlocks);
  reduce_int_tuple_kernel<<<numBlocks, threads_reduce>>>((int*)arr_1, (int*)arr_2, (int*)res_1, (int*)res_2, 
                                                   size, b_1, b_2, hof);
  reduce_int_tuple_kernel<<<1, 1024>>>((int*)res_1, (int*)res_2, (int*)res_1, (int*)res_2, numBlocks, b_1, b_2, hof);

  std::pair<int, int> ret;
  cudaMemcpy(&ret.first,  res_1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ret.second, res_2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(res_1);
  cudafree(res_2);

  return ret;
}

//BEGIN SCAN

__device__ __inline__
std::pair<int, int> warp_scan_shfl(int b_1, int b_2, scan_fun_int_tuple f, int* out_1, int* out_2, int idx, int length){
  int warpIdx = threadIdx.x % warpSize;
  std::pair<int, int> res;
  if(idx < length){
    res.first = out_1[idx];
    res.second = out_2[idx];
  }
  else{
    res.first = b_1;
    res.second = b_2;
  }
  #pragma unroll
  for(int i = 1;i < warpSize;i *= 2){
    int a_1 = __shfl_up(res.first, i);
    int a_2 = __shfl_up(res.second, i);
    if(i <= warpIdx){
      res = f(a_1, a_2, res.first, res.second);
    }
  }
  if(idx < length){
    out_1[idx] = res.first;
    out_2[idx] = res.second;
  }
  return res;
}

__device__ __inline__
std::pair<int, int> block_scan(int* in_1, int* in_2, int length, scan_fun_int_tuple f, int b_1, int b_2){

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
void scan_intxint_kernel(int* in, int* block_results, scan_fun_intxint f, int b, int length){
  
  int block_res = block_scan(in, length, f, b);
  if(threadIdx.x == block_red_size_scan - 1){
    block_results[blockIdx.x] = block_res;
  }
}
__global__
void compress_results(int* block_res, int* out, int len, scan_fun_intxint f){
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
void serial_scan(int* bres, int len, int b, scan_fun_intxint f){
  int res = b;
  #pragma unroll
  for(int i = 0;i < len;i++){
    res = f(res, bres[i]);
    bres[i] = res;
  }
}

extern "C"
void* inclusive_scan_intxint(void* in, void* f, int length, int b){
  
  scan_fun_intxint hof = (scan_fun_intxint)f;

  int num_blocks_first = (length / threads_scan) + 1;
  int* block_results;
  int* dummy;
  cudaMalloc(&block_results, sizeof(int) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(int));

  scan_intxint_kernel<<<num_blocks_first, threads_scan>>>
          ((int*)in, block_results, hof, b, length);

  if(num_blocks_first == 1){
    cudaDeviceSynchronize();
    cudaFree(block_results);
    cudaFree(dummy);
    return in;
  }
  else if(num_blocks_first <= 1024){
    scan_intxint_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
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
    scan_intxint_kernel<<<leftover, threads_scan>>>
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

__global__
void excl_compress_results(int* block_res, int* out, int len, scan_fun_intxint f, int* final, int b){
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
int exclusive_scan_intxint(void* in, void* f, int length, int b){
  
  scan_fun_intxint hof = (scan_fun_intxint)f;

  int num_blocks_first = (length / threads_scan) + 1;
  int* block_results;
  int* dummy;
  int* final_val;
  cudaMalloc(&block_results, sizeof(int) * num_blocks_first);
  cudaMalloc(&dummy, sizeof(int));
  cudaMalloc(&final_val, sizeof(int));


  scan_intxint_kernel<<<num_blocks_first, threads_scan>>>
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
    scan_intxint_kernel<<<1, 1024>>>(block_results, dummy, hof, b, num_blocks_first);
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
    scan_intxint_kernel<<<leftover, threads_scan>>>
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

__global__
void filter_map(int* in, int* out1, int len, filter_fun_intxint f){
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
void squish(int* in, int* scanned, int* out, int length, filter_fun_intxint f){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(idx < length){
    if(f(in[idx]) == 1){
      out[scanned[idx]] = in[idx];
    }
  }
}

extern "C"
void* filter_intxint(void* arr, int length, void* f, Pointer outlen){
  filter_fun_intxint hof = (filter_fun_intxint)f;
  
  int blocks = (length / threads_filter) + 1;
    
  // make buffer array

  // this map could have been fused in with the scan with some 
  // extra code copy pasta i didnt want to do

  int* scanned;
  cudaMalloc(&scanned, sizeof(int) * length);
  filter_map<<<blocks, threads_filter>>>((int*)arr, scanned, length, hof);
  
  //scan over the bits
  reduce_fun_intxint add = (reduce_fun_intxint)gen_add_intxint();
  int len = exclusive_scan_intxint(scanned, (void*)add, length, 0);

  int* res;
  cudaMalloc(&res, sizeof(int) * len);

  squish<<<blocks, threads_filter>>>((int*)arr, scanned, res, length, hof);
  *(int*)outlen = len;
  //cudaFree(bits);
  cudaFree(scanned);
  return res;
}

__global__
void zipsquish(int* arr1, int* arr2, int* out, zipwith_fun_intxint f, int length){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < length){
    out[idx] = f(arr1[idx], arr2[idx]);
  }
}

extern "C"
void* zipwith_intxint(void* arr1, void* arr2, void* f, int length){

  zipwith_fun_intxint hof = (zipwith_fun_intxint)f;
  
  int* res;
  cudaMalloc(&res, sizeof(int) * length);

  int blocks = (length / threads_filter) + 1;
  zipsquish<<<blocks, threads_filter>>>((int*)arr1, (int*)arr2, res, hof, length);

  cudaDeviceSynchronize();
  return res;
}
//Reduce - cite http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf - another reduction algorithm choice
