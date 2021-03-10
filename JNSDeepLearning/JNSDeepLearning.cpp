#include <iostream>
#include <cuda_runtime.h>

__global__ void printHelloCUDA()
{
    printf("Hello CUDA!\n");
}

int main()
{
    std::cout << "Hello World!\n";
    printHelloCUDA << <1, 1 >> > ();
}