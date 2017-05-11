#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>
#include <limits.h>
#include <time.h>
#include <chrono>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>

/*
struct sub : public thrust::binary_function< int, int, int >
{
  __host__ __device__
  int operator()(const int &a, const int &b) const
  {
    return a - b;
  }
};
*/
__global__
void kernel(int* a, int* b, int length){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(idx >= length) return;

  a[idx] = a[idx] - b[idx];
}

int main(int argc, char** argv){
  int size = atoi(argv[1]);
  
  thrust::device_vector<int> test(size);
  thrust::fill(test.begin(), test.end(), 1);
  
  auto started = std::chrono::high_resolution_clock::now();

  thrust::inclusive_scan(test.begin(), test.end(), test.begin());
  thrust::device_vector<int> mins(size);

  thrust::copy(thrust::device, test.begin(), test.end(), mins.begin());
  thrust::exclusive_scan(mins.begin(), mins.end(), mins.begin(), INT_MAX, thrust::minimum<int>());
  
  int blocks = (size / 256) + 1;

  kernel<<<blocks, 256>>>(thrust::raw_pointer_cast(test.data()), thrust::raw_pointer_cast(mins.data()),size);


  int final = thrust::reduce(test.begin(), test.end(), INT_MIN, thrust::maximum<int>());

  auto end = std::chrono::high_resolution_clock::now();
    
  printf("Thrust time %.4f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(end - started).count()) / 1000.0);
  return 0;
}
