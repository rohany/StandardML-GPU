#include "../headers/hofs.h"

__device__
int2 const1_int_tuple(int x){
  return make_int2(1,1);
}
__device__ tabulate_fun_int_tuple const1_int_tuple_dev = const1_int_tuple;

extern "C"
void* gen_const1_int_tuple(){
  tabulate_fun_int_tuple local;
  cudaMemcpyFromSymbol(&local, const1_int_tuple_dev, sizeof(tabulate_fun_int_tuple));
  return (void*)local;
}
