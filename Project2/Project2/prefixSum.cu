#include <cuda.h>
#include <cuda_runtime.h>
#include "prefixSum.h"
#include <cmath>
#include <thrust/random.h>
float a, b ,c;
#define blocksize  128
void checkCUDAError(const char *msg, int line)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

__global__ void scan(float *arr, float *result, int n){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < n){
		/*for(int offset = 1; offset < n-1; offset*2){
			if(index >= offset){
				result[index] = arr[index - offset] + arr[index];
			}
			else {
				result[index] = arr[index];
			}
			__syncthreads();

			float *temp = arr;
			arr = result;
			result = temp;
		}*/
		int logn = ceil(log(float(n))/log(2.0f));
		for (int d=1; d<=logn; d++){    
			int offset = powf(2.0f, d-1); 
			if (index >= offset){
				result[index] = arr[index-offset] + arr[index];
			}
			else{
				result[index] = arr[index]; 
			}
			__syncthreads();

			float* temp = arr;
			arr = result;
			result = temp;
		  }
	}
}


__global__ void scanSharedSingleBlock(float *arr, float *result, int n){
	extern __shared__ float temp[];
	int index = threadIdx.x;
	if(index < n){
		int pout = 0, pin = 1;
		temp[pout*n + index] = arr[index];
		__syncthreads();
		for(int offset = 1; offset < n; offset *= 2){
			pout = 1 - pout;
			pin =  1- pout;
			if(index >= offset)
				temp[pout*n + index] = temp[pin*n + index - offset] + temp[pin*n +index];
			else
				temp[pout*n + index] = temp[pin*n + index];
			__syncthreads();
		}
		result[index] = temp[pout*n + index];
	}
}

__global__ void scanSharedArbitraryLength(float *arr, float *result, int n, float* sums){
	extern __shared__ float temp[];
	
	int index = threadIdx.x;
	int globalIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	if(globalIndex < n){
		float *tempIn = &temp[0];
		float *tempOut = &temp[n];
	
		tempOut[index] = arr[globalIndex];
		__syncthreads();
		for(int offset = 1; offset < n; offset *= 2){
			float* temp = tempIn;
			tempIn = tempOut;
			tempOut = temp;
			//__syncthreads();
			if(index >= offset){
				tempOut[index] = tempIn[index - offset] + tempIn[index];
			}
			else{
				tempOut[index] = tempIn[index];
			}
			
			__syncthreads();
		}
		result[globalIndex] = tempOut[index];
		if(index == blocksize -1) 
			sums[blockIdx.x] = tempOut[index];//last element in this block
	}
}
__global__ void getIncr(float* arr, float* result, int n, int d){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < n){
		if(index >= (int)pow(2.0,d-1)){
			result[index] = arr[index - d] + arr[index];
		}
		else{
			result[index] = arr[index];
		}
		
	}
}
__global__ void addIncr(float *Incr, float *arr, int n){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < n){
		if(blockIdx.x >= 1){
			arr[index] += Incr[blockIdx.x-1];
		}
	}
}
__global__ void scatterSetup(float *arr, float *result, int n){
	__shared__ float temp[blocksize];
	__shared__ float temp2[blocksize];
	int globalIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int index = threadIdx.x;
	temp[index] = arr[globalIndex];
	__syncthreads();
	if(globalIndex < n){
		if(temp[index] == 0)
			temp2[index] = 0;
		else 
			temp2[index] = 1;
		__syncthreads();

		for(int offset = 1; offset <= blocksize; offset*=2){
			if(index >= offset){
				temp[index] = temp2[index - offset] + temp2[index];
			}
			else{
				temp[index] = temp2[index];
			}
			temp2[index] = temp[index];
			__syncthreads();
		}
		result[globalIndex] = temp2[index];
	}
}
__global__ void ScanAdd (float *arr, float *b, int size){
	__shared__ int temp[blocksize];

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	temp[threadIdx.x] = arr[index];
	__syncthreads();

	for(int b = 0; b < blockIdx.x ; ++b){
		temp[threadIdx.x] += arr[ (b + 1) * blocksize - 1];
	}
	b[index] = temp[threadIdx.x];
}
__global__ void scatterShift(float *arr, float *result, int n){
	
	//__shared__ int temp[blocksize];
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index == 0){		
		result[index] = 0;
	}
	else 
		result[index] = arr[index - 1];
	
}
__global__ void scatter(float *arr, float *arr_scan, float *result, int n){
	__shared__ float temp[blocksize];
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	temp[threadIdx.x] = arr[index];
	__syncthreads();

	if(temp[threadIdx.x] != 0){
		int newindex = (int)arr_scan[index];
		result[newindex] = temp[threadIdx.x];
	}
}
int padn(int n){
	int i;
	for(i = 1; n > i*blocksize; i++){}
	return i*blocksize;
}
void shiftRight(float * arr, int n){
	for(int i = n-1; i > 0; i--){
		arr[i] = arr[i-1];	
	}
	arr[0] = 0;
}
void scanGPU(int n, float *in_arr, float *out_arr, int type){

	//dim3 dimBlock(1, 1);//how to decide?
	//dim3 dimGrid(n, 1); 
	
	dim3 fullBlocksPerGrid((int)ceil(float(n)/float(blocksize)));
	dim3 threadsPerBlock(blocksize); 

	int size =n*sizeof(float);
	float *in_arr_d, *out_arr_d;
	cudaMalloc((void**)&in_arr_d, size);
	cudaMemcpy(in_arr_d, in_arr, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&out_arr_d, size);
	checkCUDAErrorWithLine("Kernel failed!");
	int n_round = padn(n);
	//-----------------naive global-----------------------------------------
	if(type == 1){
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0);
		for(int i = 0; i < itertimes; i++)
			scan<<<fullBlocksPerGrid, threadsPerBlock>>>(in_arr_d, out_arr_d, n_round);
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		float time = 0.0f;
		cudaEventElapsedTime( &time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		printf(" %.4f ms \n", time);
	}
	//----------------shared single block-----------------------------------
	if(type == 2){
		
		

			int sumNum = (int)ceil(float(n)/float(blocksize));
			if(sumNum < 1){
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord( start, 0);
				for(int i = 0; i < itertimes; i++){
					scanSharedSingleBlock<<<fullBlocksPerGrid, threadsPerBlock, 2*n*sizeof(float)>>>(in_arr_d, out_arr_d, n);
				}
				cudaEventRecord( stop, 0);
				cudaEventSynchronize( stop );
				float time = 0.0f;
				cudaEventElapsedTime( &time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				printf(" %.4f ms \n", time);
				checkCUDAErrorWithLine("Kernel failed!");
			}
		//----------------shared arbitrary length-------------------------------
			else{
				float *sums_d, *incr_d;
				int sumsize = sumNum * sizeof(float);
				cudaMalloc((void**)&sums_d, sumsize);
				cudaMalloc((void**)&incr_d, sumsize);
				
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord( start, 0);
				for(int i = 0; i < itertimes; i++){
					scanSharedArbitraryLength<<<fullBlocksPerGrid, threadsPerBlock, 2*n_round*sizeof(float)>>>(in_arr_d, out_arr_d, n_round, sums_d);
				}
				cudaEventRecord( stop, 0);
				cudaEventSynchronize( stop );
				float time = 0.0f;
				cudaEventElapsedTime( &time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				printf(" %.4f ms \n", time);
				checkCUDAErrorWithLine("Kernel failed!");
				int sumNum_round = padn(sumNum);
				scan<<<fullBlocksPerGrid, threadsPerBlock>>>(sums_d, incr_d, sumNum_round);

				/*dim3 sumBlocksPerGrid((int)ceil(sumNum/(float)blocksize));
				for(int d = 1; (int)pow(2.0,d-1) <= sumNum ;d++){
					getIncr<<<sumBlocksPerGrid, threadsPerBlock>>>(sums_d, incr_d, sumNum, d);
					cudaThreadSynchronize();
					float *temp = sums_d;
					sums_d = incr_d;
					incr_d = temp;
				}*/
				checkCUDAErrorWithLine("Kernel failed!");
				addIncr<<<fullBlocksPerGrid, threadsPerBlock>>>(incr_d, out_arr_d, n);
				checkCUDAErrorWithLine("Kernel failed!");
				cudaDeviceSynchronize();
				cudaFree(sums_d);
				cudaFree(incr_d);
			}
		
		
	}
	

	
	//----------------copy to host and shift------------------------------------
	cudaMemcpy(out_arr, out_arr_d, size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("Kernel failed!");
	shiftRight(out_arr, n);
	checkCUDAErrorWithLine("Kernel failed!");

	cudaFree(in_arr_d);
	cudaFree(out_arr_d);
	
}

void scatterGPU(int n, float *in_arr, float *out_arr){
	dim3 fullBlocksPerGrid((int)ceil(float(n)/float(blocksize)));
	dim3 threadsPerBlock(blocksize); 

	int size =n*sizeof(float);
	float *in_arr_d, *out_arr_d, *arr_preScan, *arr_postScan, *arr_scan;
	cudaMalloc((void**)&in_arr_d, size);
	cudaMemcpy(in_arr_d, in_arr, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&out_arr_d, size);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMalloc((void**)&arr_preScan, size);
	cudaMalloc((void**)&arr_postScan, size);
	cudaMalloc((void**)&arr_scan, size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0);
	for(int i = 0; i < itertimes; i++){
		scatterSetup<<<fullBlocksPerGrid, threadsPerBlock>>>(in_arr_d, arr_preScan, n);
		ScanAdd<<<fullBlocksPerGrid, threadsPerBlock>>>(arr_preScan, arr_postScan, n);
		scatterShift<<<fullBlocksPerGrid, threadsPerBlock>>>(arr_postScan, arr_scan, n);
		scatter<<<fullBlocksPerGrid, threadsPerBlock>>>(in_arr_d, arr_scan, out_arr_d, n);
	}
	cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop );
	float time = 0.0f;
	cudaEventElapsedTime( &time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf(" %.4f ms \n", time);
	
	checkCUDAErrorWithLine("Kernel failed!");
	//cudaDeviceSynchronize();
	cudaMemcpy(out_arr, out_arr_d, size, cudaMemcpyDeviceToHost);

	cudaFree(in_arr_d);
	cudaFree(out_arr_d);
	cudaFree(arr_preScan);
	cudaFree(arr_postScan);
	cudaFree(arr_scan);

}
