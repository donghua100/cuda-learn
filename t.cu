#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024
#define THRREADS_NUM 256

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
	const int size = DATA_SIZE/THRREADS_NUM;
	int sum = 0;
	for (int i = tid * size; i < (tid + 1)*size && i < DATA_SIZE; i++) {
		sum += nums[i] * nums[i];
	}
	result[tid] = sum;
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

	clock_t start = clock();
	int *gpudata, *result;
	cudaMalloc((void **)&gpudata, sizeof(int)*DATA_SIZE);
	cudaMalloc((void **)&result, sizeof(int)*THRREADS_NUM);
	cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<1,THRREADS_NUM,0>>>(gpudata, result);

	int sum[THRREADS_NUM];
	cudaMemcpy(&sum, result, sizeof(int)*THRREADS_NUM, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	cudaDeviceSynchronize();
	clock_t end = clock();

	int final_sum = 0;
	for (int i = 0; i < THRREADS_NUM; i++) final_sum += sum[i];

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
