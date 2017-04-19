#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>

int main(int argc, char** argv){
  int size = atoi(argv[1]);
  
  thrust::device_vector<int> test(size);
  thrust::fill(test.begin(), test.end(), 1);
  
  auto started = std::chrono::high_resolution_clock::now();
  thrust::exclusive_scan(test.begin(), test.end(), test.begin(), 0, thrust::plus<int>());
  auto end = std::chrono::high_resolution_clock::now();
    
  printf("Thrust time %.4f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(end - started).count()) / 1000.0);
  return 0;
}
