/*
    对计时函数进行评估。
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <cuda.h>
#include <windows.h>
#include <chrono>

using namespace std::chrono;

__global__ void vectorAdd(double *A, double *B, double *C, int numElements)
{
    int tid = blockIdx.x*1024+threadIdx.x;  
    if(tid<numElements)  
    {  
        C[tid] = A[tid]+B[tid];  
    }  
    __syncthreads();
}

__global__ void GPU_init()
{/*预热GPU，调用一个空的核函数*/}

/**
 * Host main routine
 */
const int N=20000000;
double a[N],b[N],c[N];  
int main(int argc, char **argv)
{
    // 预热GPU
    GPU_init<<<1, 1>>>();
    srand((unsigned)time(NULL));
    for(int i=0;i<N;i++)  
    {  
        a[i] = rand()%N;  
        b[i] = (rand()%N)*(-1);  
    } 
    for(int j=1;j<=11;j++)
    {
    cudaDeviceSynchronize();
    // 变量申请
    LARGE_INTEGER nFreq;
    LARGE_INTEGER nBeginTime,nEndTime;
    double Qtime;

    float elapsedTime = 0.0;
    cudaEvent_t event_start, event_stop;

    clock_t clock_start;
    clock_t clock_end;

    std::chrono::time_point<std::chrono::high_resolution_clock> c11_start, c11_end;

    DWORD t1,t2;

    if(atoi(argv[1]) == 1) {
        QueryPerformanceFrequency(&nFreq);
        QueryPerformanceCounter(&nBeginTime); 
    } else if(atoi(argv[1]) == 2) {
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);
        cudaEventRecord(event_start, 0);
    } else if(atoi(argv[1]) == 3) {
        clock_start = clock();
    } else if(atoi(argv[1]) == 4) {
        c11_start = high_resolution_clock::now();
    } else if(atoi(argv[1]) == 5) {
        t1 = GetTickCount();
    }

    double *dev_a , *dev_b, *dev_c;  
    cudaMalloc( (void**)&dev_a, N*sizeof(double) );  
    cudaMalloc( (void**)&dev_b, N*sizeof(double) );  
    cudaMalloc( (void**)&dev_c, N*sizeof(double) );  
    
 

    // 四种计时方式

    /*vectorAdd代码，包含内存申请、初始化、拷贝、计算、拷回、释放。数据量大小为5000000*/

    dim3 dimBlock(ceil((double)N/1024.0));
    dim3 dimGrid(1024);
    cudaMemcpy( dev_a , a, N*sizeof(int), cudaMemcpyHostToDevice ) ;  
    cudaMemcpy( dev_b , b, N*sizeof(int), cudaMemcpyHostToDevice ) ;  
    vectorAdd<<<dimBlock,dimGrid>>>(dev_a,dev_b,dev_c,N);
    cudaMemcpy( c , dev_c , N*sizeof(int), cudaMemcpyDeviceToHost) ;  
    cudaFree(dev_a);  
    cudaFree(dev_b);  
    cudaFree(dev_c);  

    if(atoi(argv[1]) == 1) {
        // 如果使用CPU计时方式，一定要加同步函数！！
        cudaDeviceSynchronize();
        QueryPerformanceCounter(&nEndTime);
        Qtime=(double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
        printf("QueryPerformanceCounter time = %fms\n", Qtime*1000);
    } else if(atoi(argv[1]) == 2) {
        cudaDeviceSynchronize();
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
        printf("cudaevent time = %lfms\n", elapsedTime);
    } else if(atoi(argv[1]) == 3) {
        cudaDeviceSynchronize();
        clock_end= clock();
        double clock_diff_sec = ((double)(clock_end- clock_start) / CLOCKS_PER_SEC);
        printf("clock_ time: %lfms.\n", clock_diff_sec * 1000);
    }else if(atoi(argv[1]) == 4) {
        cudaDeviceSynchronize();
        c11_end = high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>
                             (c11_end-c11_start).count();
        printf("chrono time: %lfms.\n", elapsed_seconds/1000/1000);
    }  else if(atoi(argv[1]) == 5) {
        t2 = GetTickCount();
        printf("GetTick time: %lfms.\n", double(t2-t1));
    }
    //printf("done!\n");
    }
    return EXIT_SUCCESS;
}