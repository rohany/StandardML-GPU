#include "../headers/hofs.h"

__device__ __inline__
float add_float(float x, float y){
  return x+y;
}
__device__ reduce_fun_float add_dev_float = add_float;
extern "C"
void* gen_add_float(){
  reduce_fun_float local;
  cudaMemcpyFromSymbol(&local, add_dev_float, sizeof(reduce_fun_float));
  return (void*)local;
}
