#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#ifdef __cplusplus
extern "C" {
#endif

	class CUDA_Matrix
	{
	public:
		CUDA_Matrix();
		virtual ~CUDA_Matrix();

		void Matrix_Multiply(const double* _A, const double* _B, double* _Result, int _M, int _N, int _K);
		void Matrix_Multiply_Async(const double* _A, const double* _B, double* _Result, int _M, int _N, int _K);
	};
#ifdef __cplusplus
}
#endif