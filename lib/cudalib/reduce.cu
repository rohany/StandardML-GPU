#include <cuda.h>
#include "hofs.h"
#include "export.h"

/*
__inline__ __device__
int warp_red_int(int b, reduce_fun_int f){
  int res = b;
  for(int i = warpSize / 2;i >= 1;i /= 2){
    int a = __shfl_down(res, i);
    res = (*f)(res, a);
  }
  return res;
}
*/

extern "C"
int reduce_int(void* arr, int size, int b, void* f){
  reduce_fun_int hof = (reduce_fun_int)f;

  return 0;
}
