#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust\host_vector.h>
#include <ctime>
#include "PrefixSum.h"
using namespace std;


int n = 5000;
#define GLOBAL 1
#define SHARED 2

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
void thrustScatter(float *arr, int n){
	thrust::host_vector<int> h_input(n);
	thrust::host_vector<int> h_input_bool(n);
	thrust::host_vector<int> h_map(n);
	thrust::host_vector<int> h_output(n);
	for(int i = 0; i < n ; ++i){
		h_input[i] = arr[i];
	}
	for(int i = 0; i < n; i++){
		if(h_input[i] != 0){
			h_input_bool[i] = 1;
		}		
	}
	thrust::exclusive_scan(h_input_bool.begin(), h_input_bool.end(), h_map.begin());
	thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
}
void main(){
	//-----------------test case setup-----------------------
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
	clock_t begin = clock();
	for(int i = 0; i < itertimes; i++){
		scatterCPU(in_arr, out_arr, n, length);
	}
	clock_t end = clock();
	double time = (end - begin)/(CLOCKS_PER_SEC / 1000.0);
	printf(" %.4f ms \n", time);
	//printArr(length, out_arr);
	scatterGPU(n, in_arr, out_arr2);
	//printArr(length, out_arr2);
	//-------------scan----------------
	//clock_t begin = clock();
	//for(int i = 0; i < itertimes; i++){
	prefixSumCPU(in_arr, out_arr, n);
	////printArr(n, out_arr);
	//}
	//clock_t end = clock();
	//double time = (end - begin)/(CLOCKS_PER_SEC / 10000.0);
	//printf(" %.4f ms \n", time);

	
	scanGPU(n, in_arr, out_arr2, SHARED);
	scanGPU(n, in_arr, out_arr2, GLOBAL);
	//printArr(n, out_arr);
	
	
	//printArr(n, out_arr2);
	
	
	
}