#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>


struct generator{
  __host__ __device__
  int operator() (const int& i) const{
    return i % 2 == 0 ? 1 : -1;
  }  
};

int main(int argc, char** argv){
  int size = atoi(argv[1]);
  
  thrust::device_vector<int> test(size);
  thrust::tabulate(test.begin(), test.end(), generator());
  
  auto started = std::chrono::high_resolution_clock::now();
  thrust::inclusive_scan(test.begin(), test.end(), test.begin(), thrust::plus<int>());
  int last = test[test.size() - 1];
  int min = thrust::reduce(test.begin(), test.end(), 0, thrust::minimum<int>());
  bool matched = last == 0 && min >= 0;
  auto end = std::chrono::high_resolution_clock::now();
    
  printf("Thrust time %.4f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(end - started).count()) / 1000.0);
  return 0;
}
