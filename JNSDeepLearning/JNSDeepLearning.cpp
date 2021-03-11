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
    printHelloCUDA << <1, 1 >> > (); //<<<그리드 당 블록 수, 블록당 스레드 수>>>
    
    float *A, *B, *C; //디바이스 메모리 - GPU 쪽 메모리 
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
    MatrixAdd<<<1, ThreadPBlock>>>(A, B, C); //블록도 dim3로 정의할 수 있음, 블록을 묶은 것을 '그리드'라고 부름.
    
    cudaMemcpy(c, C, N*N*sizeof(*C), cudaMemcpyHostToDevice);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
