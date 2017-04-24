#include "../headers/hofs.h"

__device__
bool is_even_int(int x){
  return (x & 1) == 0;
}
__device__ filter_fun_int is_even_int_dev = is_even_int;
extern "C"
void* gen_is_even_int(){
  filter_fun_int local;
  cudaMemcpyFromSymbol(&local, is_even_int_dev, sizeof(filter_fun_int));
  return (void*)local;
}
