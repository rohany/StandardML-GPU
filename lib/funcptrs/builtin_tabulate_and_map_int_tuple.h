#include "../headers/hofs.h"

__device__
int2 const1_int_tuple(int x){
  return make_int2(1,1);
}
__device__ tabulate_fun_int_tuple identity_int_tuple_dev = identity_int_tuple;

__device__
int2 identity_int_tuple(int x){
  return make_int2(1,1);
}
__device__ tabulate_fun_int_tuple identity_int_tuple_dev = identity_int_tuple;