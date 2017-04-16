#include "../headers/export.h"
#include "../headers/hofs.h"

// basic builtin hofs, can extend this later

// i believe this should work even if we let it be 
// the map operator type
__device__
int identity(int x){
  return x;
}
__device__ tabulate_fun_int identity_dev = identity;
extern "C"
void* gen_identity(){
  tabulate_fun_int local;
  cudaMemcpyFromSymbol(&local, identity_dev, sizeof(tabulate_fun_int));
  return (void*)local;
}

__device__
int _double(int x){
  return x;
}
__device__ tabulate_fun_int double_dev = _double;
extern "C"
void* gen_double(){
  tabulate_fun_int local;
  cudaMemcpyFromSymbol(&local, double_dev, sizeof(tabulate_fun_int));
  return (void*)local;
}

