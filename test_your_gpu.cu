#include<stdio.h>
#include<string.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <iomanip>
#include <time.h>
using namespace std;
int main(){
	int dev = 0;
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    cout << "GPU device: " << dev << ": " << devProp.name << std::endl;
    cout << "number of SM: " << devProp.multiProcessorCount << std::endl;
    cout << "The Shared memory size of each thread block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    cout << "Maximum number of threads per thread block: " << devProp.maxThreadsPerBlock << std::endl;
    cout << "Maximum number of threads per EM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    cout << "Maximum number of thread bundles per EM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    return 0;
}

