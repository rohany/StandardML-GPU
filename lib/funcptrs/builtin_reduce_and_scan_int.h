#include "../headers/hofs.h"

__device__ __inline__
int add_int(int x, int y){
  return x+y;
}
__device__ reduce_fun_int add_dev_int = add_int;
extern "C"
void* gen_add_int(){
  reduce_fun_int local;
  cudaMemcpyFromSymbol(&local, add_dev_int, sizeof(reduce_fun_int));
  return (void*)local;
}

__device__ __inline__
int left_int(int x, int y){
  return x;
}
__device__ reduce_fun_int left_dev_int = left_int;
extern "C"
void* gen_left_int(){
  reduce_fun_int local;
  cudaMemcpyFromSymbol(&local, left_dev_int, sizeof(reduce_fun_int));
  return (void*)local;
}
