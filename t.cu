#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024

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
	int sum = 0;
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += nums[i] * nums[i];
	}
	*result = sum;
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
	cudaMalloc((void **)&result, sizeof(int));
	cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<1,1,0>>>(gpudata, result);

	int sum;
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);

	cudaDeviceSynchronize();
	clock_t end = clock();


	printf("(GPU) sum = %d, using time: %lf ms\n", sum, (double)(end - start)/CLOCKS_PER_SEC*1000);

	sum = 0;
	start = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i];
	}
	end = clock();
	// printf("(CPU) sum = %d, using time: %ld\n", sum, time_used);
	printf("(CPU) sum = %d, using time: %lf ms\n", sum, (double)(end - start)/CLOCKS_PER_SEC*1000);

	return 0;
}
