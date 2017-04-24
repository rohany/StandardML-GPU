#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>

struct is_even{
  __host__ __device__
  bool operator () (const int x){
    return (x & 1) == 0;
  }
};

int main(int argc, char** argv){
  int size = atoi(argv[1]);
  
  thrust::device_vector<int> test(size);
  thrust::fill(test.begin(), test.end(), 1);
  
  auto started = std::chrono::high_resolution_clock::now();
  thrust::device_vector<int> res(size);
  thrust::copy_if(test.begin(), test.end(), res.begin(), is_even());
  auto end = std::chrono::high_resolution_clock::now();
    
  printf("Thrust time %.4f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(end - started).count()) / 1000.0);
  return 0;
}
