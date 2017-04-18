#include "../headers/hofs.h"

__device__
inline int add_int(int x, int y){
  return x+y;
}
__device__ reduce_fun_int add_dev_int = add_int;
extern "C"
void* gen_add_int(){
  reduce_fun_int local;
  cudaMemcpyFromSymbol(&local, add_dev_int, sizeof(reduce_fun_int));
  return (void*)local;
}
