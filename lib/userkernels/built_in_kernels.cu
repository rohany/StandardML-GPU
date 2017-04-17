#include "../headers/export.h"
#include "../headers/hofs.h"

// basic builtin hofs, can extend this later

// i believe this should work even if we let it be 
// the map operator type
__device__
int identity_int(int x){
  return x;
}
__device__ tabulate_fun_int identity_dev_int = identity_int;
extern "C"
void* gen_identity_int(){
  tabulate_fun_int local;
  cudaMemcpyFromSymbol(&local, identity_dev_int, sizeof(tabulate_fun_int));
  return (void*)local;
}

__device__
int _double_int(int x){
  return x;
}
__device__ tabulate_fun_int double_dev_int = _double_int;
extern "C"
void* gen_double_int(){
  tabulate_fun_int local;
  cudaMemcpyFromSymbol(&local, double_dev_int, sizeof(tabulate_fun_int));
  return (void*)local;
}

__device__
int add_int(int x, int y){
  return x + y;
}
__device__ reduce_fun_int add_dev_int = add_int;
extern "C"
void* gen_add_int(){
  reduce_fun_int local;
  cudaMemcpyFromSymbol(&local, add_dev_int, sizeof(reduce_fun_int));
  return (void*)local;
}
