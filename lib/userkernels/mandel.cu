#include <stdio.h>
#include <time.h>
#include "export.h"

__global__
void mandel(int* out, int count, int width, int height, 
            float dx, float dy, float x0, float y0){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= height * width){
    return;
  }

  float c_re = x0 + (idx % width) * dx;
  float c_im = y0 + (idx / width) * dy;

  float z_re = c_re;
  float z_im = c_im;

  if(idx == 1000){
    printf("%.9f %.9f %.9f %.9f\n", z_re, z_im, c_re, c_im);
  }
  int i;
  for(i = 0;i < count;i++){
    if(z_re * z_re + z_im * z_im > 4.f){
      break;
    }
    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }
  if(idx == 1000){
    printf("%f %f %f %f\n", z_re, z_im, c_re, c_im);
  }

  out[idx] = i;
  if(idx == 1000){
    printf("hi %d\n", i);
  }
}

extern "C"
void mandel_gpu(void* out, int count, int width, int height,
                float dx, float dy, float x0, float y0){
  int blocks = ((width * height) / 256) + 1;
  int threads = 256;
  mandel<<<blocks, threads>>>((int*)out, count, width, height, dx, dy, x0, y0);
  printf("height %d width %d\n", height, width);
  cudaDeviceSynchronize();
}
