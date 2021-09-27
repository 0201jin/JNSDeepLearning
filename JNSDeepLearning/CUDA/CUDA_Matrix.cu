#include "CUDA_Matrix.cuh"

__global__ void kernel(void) { }

__global__ void CUDA_Matrix_Multiply(const double* _A, const double* _B, double* _Result, int _M, int _N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    double value = 0;
    for (int k = 0; k < _N; ++k)
        value += _A[k * _M + j] * _B[i * _M + k];

	_Result[i * _N + j] = value;
}

CUDA_Matrix::CUDA_Matrix()
{
}

CUDA_Matrix::~CUDA_Matrix()
{
}

void CUDA_Matrix::Matrix_Multiply(const double* _A, const double* _B, double* _Result, int _M, int _N, int _K)
{
	dim3 grid(1, 1), block(_M, _K);

	CUDA_Matrix_Multiply << <grid, block >> > (
		_A,
		_B,
		_Result,
		_M, _N);

	cudaDeviceSynchronize();
}

void CUDA_Matrix::Matrix_Multiply_Async(const double* _A, const double* _B, double* _Result, int _M, int _N, int _K)
{
	dim3 grid(1, 1), block(_M, _K);

	CUDA_Matrix_Multiply << <grid, block >> > (
		_A,
		_B,
		_Result,
		_M, _N);
}