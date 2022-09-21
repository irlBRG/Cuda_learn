#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include<iostream>
#include <stdio.h>
#include <chrono>
#include<time.h>

#define N 128 //количество элементов в массиве

void add_cpu(int* a, int* b, int* c)
{
    int tid = 0;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

__global__ void add_gpu(int* a, int* b, int* c)
{
    int tid = threadIdx.x; // индекс элемента
    if (tid > N - 1)// проверка за пределы массива
        return;// 
    c[tid] = a[tid] + b[tid];// сложение массивов
}
int main()
{
    int a[N];// память на CPU
    int b[N];
    int c[N];


    for (int i = 0; i < N; i++) //заполнение массива
    {
        a[i] = -i;
        b[i] = i * i;
    }

    long before = time(NULL);
    add_cpu(a, b, c);
    long after = time(NULL);


    for (int i = 0; i < N; i++)
    {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }

    int* dev_a; // память на GPU
    int* dev_b;
    int* dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(int));// память на GPU
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice); //копирование данных GPU
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_gpu << <1, N >> > (dev_a, dev_b, dev_c); //вызов ядра

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time on gpu=%f ms\n", gpuTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);//результат с GPU
    for (int i = 0; i < N; i++)// вывод рассчетов
    {
        printf("%d+%d=%d\n", a[i], b[i], c[i]);
    }
    printf("time on gpu=%.10f ms\n", gpuTime);// время GPU
    printf("time on cpu=%ld ms\n", after- before); //время CPU неудалось
    cudaFree(dev_a);//освобождение памяти GPU
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}