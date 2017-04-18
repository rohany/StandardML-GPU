#include <thrust/device_vector.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv){
  int size = atoi(argv[1]);
  
  thrust::device_vector<int> test(size);
  thrust::fill(test.begin(), test.end(), 1);
  
  clock_t begin = clock();
  thrust::inclusive_scan(test.begin(), test.end(), test.begin(), thrust::plus<int>());
  clock_t end = clock();
  printf("time spent : %.4f\n", (double) (end - begin) / CLOCKS_PER_SEC);
  return 0;
}
