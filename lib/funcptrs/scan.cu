#include "../headers/hofs.h"
#include "../headers/export.h"


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

//A will be the partialResult ( off by the sum of the elements before its warp)
//partialSums will be the last element in each partial result ( warp sum)
__global__ void k_scan_warp(int* a, int* warpSums)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int lane = idx & 31;

	if (lane >= 1) a[idx] = (a[idx - 1] + a[idx]);
	if (lane >= 2) a[idx] = (a[idx - 2] + a[idx]);
	if (lane >= 4) a[idx] = (a[idx - 4] + a[idx]);
	if (lane >= 8) a[idx] = (a[idx - 8] + a[idx]);
	if (lane >= 16) a[idx] = (a[idx - 16] + a[idx]);

	__syncthreads();

	//if (lane > 0)
	//	a[idx] = a[idx - 1];
	//else
	//	a[idx] = 0;

	if (lane == 31)
		warpSums[idx >> 5] = a[idx];
}

__global__ void k_scan_block(int maxIdx, int* a, int* blockSums)
{
	const unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	const unsigned int lane = idx & 31;

	if (idx < maxIdx)
	{
		if (lane >= 1) a[idx] = (a[idx - 1] + a[idx]);
		if (lane >= 2) a[idx] = (a[idx - 2] + a[idx]);
		if (lane >= 4) a[idx] = (a[idx - 4] + a[idx]);
		if (lane >= 8) a[idx] = (a[idx - 8] + a[idx]);
		if (lane >= 16) a[idx] = (a[idx - 16] + a[idx]);
	}

	__syncthreads();

	if ((lane == 31) && (idx < maxIdx))
		blockSums[idx >> 5] = a[idx];
}

__global__ void k_scan_sum(const int numBlocks, int* blockSums)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = 1; i < numBlocks; i <<= 1)
	{
		if (idx >= i)
		{
			blockSums[idx] = (blockSums[idx] + blockSums[idx - i]);
		}
		__syncthreads();
	}
}

__global__ void k_increment_32(int maxIdx, int* partialResults, int* increment)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int incId = idx >> 5;
	if (incId > 0)// && idx < maxIdx)
		partialResults[idx] += increment[incId - 1];
}

__global__ void k_inclusive_to_exclusive(const int n, int* input, int* inclusiveResult)
{
	const unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if (idx < n)
		inclusiveResult[idx] -= input[idx];
}

__global__ void k_labelDups(int length, int* src, int* dst)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < length - 1)
	{
		if (src[idx] == src[idx + 1])
			dst[idx] = 1; 
	}
	else if (idx == length - 1)
	{
		dst[idx] = false;
	}
}

__global__ void k_filter(int length, int* labels, int* cum_sum)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < length)
	{
		if (labels[idx])
		{
			cum_sum[cum_sum[idx]] = idx;
		}

void exclusive_scan(int* d_start, int length, int* d_result)
{
	int N = nextPow2(length);
	N = max(N, 1024);

	// compute number of blocks and threads per block
	const int warpSize = 32;
	const int threadsPerBlock = warpSize * warpSize; //= 1024;
	const int numWarps = (N >> 5);
	const int blocks  = N / threadsPerBlock;
	const int sBlocks = max(blocks / 32, 1);
	const int warpblocks = (blocks + warpSize -1) / warpSize;
	const int blockblocks = (sBlocks + warpSize - 1) / warpSize;
	

	//Unroll 3 iterations designed for 100% occupancy for large problem-sets with a
	//small constant penalty for memory allocation at the beginning

 	int* d_warpSums;
	cudaMalloc(&d_warpSums, numWarps * sizeof(int));

	int* d_blockSums;
	cudaMalloc(&d_blockSums, blocks * sizeof(int));

	int* d_superBlockSums;
	cudaMalloc(&d_superBlockSums, sBlocks * sizeof(int));

	//Scan each warp (N)
	k_scan_warp <<<blocks, threadsPerBlock >>>(d_result, d_warpSums);

	cudaDeviceSynchronize();

	//Scan each block of warp sums (N/32)
	k_scan_block <<<warpblocks, threadsPerBlock >>> (N / 32, d_warpSums, d_blockSums);

	cudaDeviceSynchronize();

	//Scan superBlock of block sums (N/1024)
	k_scan_block <<<blockblocks, threadsPerBlock >>> (N / 1024, d_blockSums, d_superBlockSums);

	cudaDeviceSynchronize();

	//Scan all block sums (N/1024)
	k_scan_sum <<<1, sBlocks >>> (sBlocks, d_superBlockSums);

	cudaDeviceSynchronize();

	k_increment_32 <<<blockblocks, threadsPerBlock >>> (N / 1024, d_blockSums, d_superBlockSums);

	cudaDeviceSynchronize();

	k_increment_32 <<<warpblocks, threadsPerBlock >>> (N / 32, d_warpSums, d_blockSums);

	cudaDeviceSynchronize();

	//Add each warp sum to the corresponding warp partial
	k_increment_32 <<<blocks, threadsPerBlock>>> (length, d_result, d_warpSums) ;

	cudaDeviceSynchronize();

	//Make exclusive scan
	k_inclusive_to_exclusive <<<blocks, threadsPerBlock >>> (length, d_start, d_result);

	cudaDeviceSynchronize();

	cudaFree(d_warpSums);
	cudaFree(d_blockSums);
	cudaFree(d_superBlockSums);                                                                                        
}


void scan_excl_int(void* arr, int size, int b, void* f){

}