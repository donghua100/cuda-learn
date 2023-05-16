#include <stdio.h>
#include <stdlib.h>
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

__global__ static void sumOfSquares(int *nums, int *result, clock_t *time) {
	int sum = 0;
	clock_t start = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += nums[i] * nums[i];
	}
	*result = sum;
	*time = clock() - start;
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
	clock_t *time;
	cudaMalloc((void **)&gpudata, sizeof(int)*DATA_SIZE);
	cudaMalloc((void **)&result, sizeof(int));
	cudaMalloc((void **)&time, sizeof(clock_t));
	cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<1,1,0>>>(gpudata, result, time);

	int sum;
	clock_t time_used;
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	printf("sum = %d, using time: %ld\n", sum, time_used);

	sum = 0;
	// clock_t start = clock();
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i];
	}
	// time_used = clock() - start;
	// printf("(CPU) sum = %d, using time: %ld\n", sum, time_used);
	printf("(CPU) sum = %d\n", sum);

	return 0;
}
