#include "export.h"

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
	cudaMemset(ret_ptr, 0, typesize * size);
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
}

extern "C"
void free_gpu_ptr(void* ptr){
	cudaFree(ptr);
}
