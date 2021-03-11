#include "Perceptron.h"

float Perceptron::Dot(float* _v1, float* _v2, int _len)
{
	float fSum = 0;

	for (int i = 0; i < _len; i++)
		fSum += _v1[i] * _v2[i];

	return fSum;
}

float Perceptron::Step(float _v)
{
	return _v > 0 ? 1 : 0;
}

float Perceptron::Forward(float* _x, float* _w, int _len)
{
	float u = Dot(_x, _w, _len);
	return Step(u);
}

float Perceptron::Train(float* _w, float* _x, float _t, float _e, int _len)
{
	float fZ = Forward(_x, _w, _len);

	for (int i = 0; i < _len; i++)
	{
		_w[i] += (_t - fZ) * _x[i] * _e;
	}

	return 0.0f;
}
