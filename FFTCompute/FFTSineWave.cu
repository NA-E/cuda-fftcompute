#include <iostream>
#include "cuda_runtime.h"
#include <cufft.h>
#include <stdio.h>

const double PI = 3.141592653589793238460;

int main() {
	
	int N = 10;
	size_t memorySize = N * sizeof(cufftComplex);

	cufftComplex* A = (cufftComplex*) malloc(memorySize);
	cufftComplex* B = (cufftComplex*)malloc(memorySize);

	for (int i = 0; i < N; i++) {
		A[i].x = (float) sin(2 * PI * i / N);
		A[i].y = 0.0;
	}

	cufftComplex* d_A;
	cufftComplex* d_B;

	cudaMalloc(&d_A, memorySize);
	cudaMalloc(&d_B, memorySize);

	cudaMemcpy(d_A, A, memorySize, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_C2C, 1);
	cufftExecC2C(plan, d_A, d_B, CUFFT_FORWARD);

	cudaMemcpy(B, d_B, memorySize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%f %f %f %f\n", A[i].x, A[i].y, B[i].x, B[i].y);
	}

	return 0;
}