﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 10

//__global__ void addKernel(int *a,int *b, int *c)
//{
   // int tid = blockIdx.x;
   // if(tid < N)
        //c[tid] = a[tid] + b[tid]
//}

void add_cpu(int *a, int *b, int *c)
{
    int tid = 0;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

__global__ void add_gpu(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
int main()
{
    int a[N];
    int b[N];
    int c[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    
    add_cpu(a, b, c);
    
    for (int i = 0; i < N; i++)
    {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    
    int *dev_a;
    int *dev_b;
    int *dev_c;
    
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add_gpu <<<N, 1 >> >(dev_a, dev_b, dev_c);

    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < N; i++)
    {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
