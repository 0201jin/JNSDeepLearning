#pragma once
#include <iostream>

/*
* ´Ü¼ø ÆÛ¼ÁÆ®·Ð
*/
namespace Perceptron
{
	float Dot(float* _v1, float* _v2, int _len);
	float Step(float _v);
	float ReLU(float _v);
	float Forward(float *_x, float *_w, int _len);
	float Train(float *_w, float *_x, float _t, float _e, int _len);
};

