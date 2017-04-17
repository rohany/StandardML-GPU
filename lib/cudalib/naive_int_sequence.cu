#include "../headers/export.h"
#include "../headers/hofs.h"

#define blockSize = 256

//Tabulate
__global__ 
void tabulate_int_kernel(int* arr, int len, tabulate_fun_int f){
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= len){
    return;
  }

  arr[idx] = (*f)(idx);

}

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

  arr[idx] = (*f)(arr[idx]);

}

void* map_int(void* inarr, void* f, int size){
  
  map_fun_int hof = (map_fun_int)f;      
  int blockNum = (size / 256) + 1;
  
  map_int_kernel<<<blockNum, 256>>>((int*)inarr, size, hof);

  return inarr;

}

//Reduce - cite http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__global__
void reduce_int_kernel(int* arr, int len, map_fun_int f){
  
  extern __shared__ int sdata[];

  unsigned int thread_id = threadIdx.x;
  unsigned int array_id = blockIdx.x * (blockSize * 2) + thread_id;
  unsigned int gridSize = 2 * blockSize * gridDim.x;
  sdata[thread_id] = 0;

  while (array_id< len) 
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
