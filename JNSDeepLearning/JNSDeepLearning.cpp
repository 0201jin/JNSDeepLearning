#include <iostream>
#include <cuda_runtime.h>

__global__ void printHelloCUDA()
{
    printf("Hello CUDA!\n");
}

int main()
{
    std::cout << "Hello World!\n";
    printHelloCUDA << <1, 1 >> > (); //1개 블럭을 생성하고 블럭마다 1개의 쓰레드를 생성한다.
}
