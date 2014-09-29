#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>




#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
void runCUDA(int n, float *in_arr, float *out_arr);
void checkCUDAError(const char *msg, int line = -1);
void shiftRight(float * arr, int n);

void scatterGPU(int n, float *in_arr, float *out_arr);
#endif