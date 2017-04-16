#include "export.h"
#include <stdio.h>

extern "C"
void init_gpu(){
  cudaDeviceSynchronize();
}

extern "C"
void* allocate_on_gpu(size_t size, int smltype){
	size_t typesize;
	if(smltype == 0){
		typesize = sizeof(int);
	}
	else{
		typesize = sizeof(float);
	}

	void* ret_ptr;
	cudaMalloc(&ret_ptr, typesize * size);
	return ret_ptr;
}

extern "C"
void* copy_float_into_gpu(Pointer src, int size){
	
	void* ret_ptr;
	cudaMalloc(&ret_ptr, sizeof(float) * size);
  cudaMemcpy(ret_ptr, src, sizeof(float) * size, cudaMemcpyHostToDevice);
  
  return ret_ptr;
}

extern "C"
void* copy_int_into_gpu(Pointer src, int size){
	
	void* ret_ptr;
	cudaMalloc(&ret_ptr, sizeof(int) * size);
  cudaMemcpy(ret_ptr, src, sizeof(int) * size, cudaMemcpyHostToDevice);
  
  return ret_ptr;
}

extern "C"
void copy_float_gpu(Pointer dest, void* gpuarr, size_t size){
	size_t typesize = sizeof(float);
  cudaMemcpy(dest, gpuarr, size * typesize, cudaMemcpyDeviceToHost);
}

extern "C"
void copy_int_gpu(Pointer dest, void* gpuarr, size_t size){
	size_t typesize = sizeof(int);
  cudaMemcpy(dest, gpuarr, size * typesize, cudaMemcpyDeviceToHost);
  printf("finished transfer\n");
}

extern "C"
void free_gpu_ptr(void* ptr){
	cudaFree(ptr);
}

__global__
void initwith_int(int* arr, int b, int len){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < len){
    arr[idx] = b;
  }
}
extern "C"
void initInt_gpu(int size, int b){
  void* dev_ptr;
  cudaMalloc(&dev_ptr, sizeof(int) * size);

  int blocks = (size / 256) + 1;
  initwith_int<<<blocks, 256>>>((int*)dev_ptr, b, size);
}

__global__
void initwith_float(float* arr, float b, int len){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < len){
    arr[idx] = b;
  }
}
extern "C"
void initFloat_gpu(int size, float b){
  void* dev_ptr;
  cudaMalloc(&dev_ptr, sizeof(float) * size);

  int blocks = (size / 256) + 1;
  initwith_float<<<blocks, 256>>>((float*)dev_ptr, b, size);
}

extern "C"
void* copy(void* in, int size, int smltype){
	size_t typesize;
	if(smltype == 0){
		typesize = sizeof(int);
	}
	else{
		typesize = sizeof(float);
	}

	void* ret_ptr;
	cudaMalloc(&ret_ptr, typesize * size);
  cudaMemcpy(ret_ptr, in, typesize * size, cudaMemcpyDeviceToDevice);
  return ret_ptr;
}

