#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024
#define THRREADS_NUM 256
#define BLOCK_NUM 32

int data[DATA_SIZE];

void generate_nums(int *nums, int size) {
	for (int i = 0; i < size; i++) {
		nums[i] = rand() %	10;
	}
}


int init_cuda() {
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device\n");
		return -1;
	}
	printf("There are %d device.\n", count);
	int i;
	for (i = 0; i < count; i++) {
		struct cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) break;
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return -1;
	}
	cudaSetDevice(i);
	return 0;
}

__global__ static void sumOfSquares(int *nums, int *result) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;


	int sum = 0;
	for (int i = bid * THRREADS_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THRREADS_NUM) {
		sum += nums[i] * nums[i];
	}
	result[bid * THRREADS_NUM + tid] = sum;
}

int main() {
	if (init_cuda() == 0) {
		printf("CUDA initialized.\n");
	}
	else {
		printf("initialized CUDA fail!\n");
		return -1;
	}
	generate_nums(data, DATA_SIZE);

	int *gpudata, *result;
	cudaMalloc((void **)&gpudata, sizeof(int)*DATA_SIZE);
	cudaMalloc((void **)&result, sizeof(int)*THRREADS_NUM*BLOCK_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

	clock_t start = clock();
	sumOfSquares<<<BLOCK_NUM,THRREADS_NUM,0>>>(gpudata, result);
	cudaDeviceSynchronize();
	clock_t end = clock();

	int sum[THRREADS_NUM * BLOCK_NUM];
	cudaMemcpy(&sum, result, sizeof(int)*THRREADS_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);


	int final_sum = 0;
	for (int i = 0; i < THRREADS_NUM * BLOCK_NUM; i++) final_sum += sum[i];

	printf("(GPU) sum = %d, using time: %lf ms\n", final_sum, (double)(end - start)/CLOCKS_PER_SEC*1000);

	final_sum = 0;
	start = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		final_sum += data[i] * data[i];
	}
	end = clock();
	// printf("(CPU) sum = %d, using time: %ld\n", sum, time_used);
	printf("(CPU) sum = %d, using time: %lf ms\n", final_sum, (double)(end - start)/CLOCKS_PER_SEC*1000);

	return 0;
}
