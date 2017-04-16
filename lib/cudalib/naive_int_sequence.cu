#include "../headers/export.h"
#include "../headers/hofs.h"

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
