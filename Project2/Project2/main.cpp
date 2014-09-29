#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "PrefixSum.h"
using namespace std;


int n = 100;
void prefixSumCPU(float* arr, float *result, int n){	
	for(int i = 1; i < n; i++){
		result[i] = arr[i-1] + result[i-1];
	}
}
void scatterCPU(float* arr, float* result, int n, int &length){
	float* arrcopy = new float[n];
	for(int i = 0; i < n; i++){
		arrcopy[i] = arr[i];
	}
	int len = 0;
	for(int i = 0; i < n; i++){
		if(arrcopy[i] != 0){
			arrcopy[i] = 1;
			len++;
		}
	}
	float* arrAfterScan = new float[n];
	arrAfterScan[0] = 0;
	prefixSumCPU(arrcopy, arrAfterScan, n);

	for(int i = 0; i < n; i++){
		if(arrcopy[i] == 1){
			int newindex = int(arrAfterScan[i]);
			result[newindex] = arr[i];
		}
	}
	delete[] arrcopy;
	delete[] arrAfterScan;
	length = len;
}


void printArr(int n, float* arr){
	for(int i = 0; i < n; i++){
		std::cout<<i<<":"<<arr[i];std::cout<<" ";
	}
	std::cout<<"\n";
}

void main(){
	//-----------------test case---------------------------
	
	
	//-----------------test case end-----------------------
	float *in_arr = new float[n];
	float *out_arr = new float[n];
	float *out_arr2 = new float[n];
	for(int i = 0; i < n; i++){
		if(i < 10)
			in_arr[i] = i;
		else in_arr[i] = i%10;
	}
	out_arr[0] = 0;
	out_arr2[0] = 0;
	//printArr(n, in_arr);
	int length = 0;
	//--------------scatter---------------
	/*scatterCPU(in_arr, out_arr, n, length);
	printArr(length, out_arr);
	scatterGPU(n, in_arr, out_arr2);
	printArr(length, out_arr2);*/
	//-------------scan----------------
	prefixSumCPU(in_arr, out_arr, n);
	printArr(n, out_arr);
	
	runCUDA(n, in_arr, out_arr2);
	printArr(n, out_arr2);
	
	
	int stop = 6;
}