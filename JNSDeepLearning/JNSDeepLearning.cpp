#include <iostream>
#include <cuda_runtime.h>

__global__ void printHelloCUDA()
{
    printf("Hello CUDA!\n");
}

__global__ void MatrixAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    std::cout << "Hello World!\n";
    printHelloCUDA << <1, 1 >> > (); //1개 블럭을 생성하고 블럭마다 1개의 쓰레드를 생성한다.
    
    float *A, *B, *C;
    int N = 50;
    cudaMalloc((void**)&A, N*N*sizeof(float));
    cudaMalloc((void**)&B, N*N*sizeof(float));
    cudaMalloc((void**)&C, N*N*sizeof(float));
    
    float *a = malloc(N*N*sizeof(float));
    float *b = malloc(N*N*sizeof(float));
    float *c = malloc(N*N*sizeof(float));
    
    cudaMemcpy(A, a, N*N*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, N*N*sizeof(*B), cudaMemcpyHostToDevice);
    
    dim3 ThreadsPBlock(N, N);
    MatrixAdd<<<1, ThreadPBlock>>>(A, B, C);
    
    cudaMemcpy(c, C, N*N*sizeof(*C), cudaMemcpyHostToDevice);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
