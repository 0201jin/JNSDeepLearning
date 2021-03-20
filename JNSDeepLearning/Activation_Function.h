#pragma once
#include <iostream>

using namespace std;

namespace Activation_Function
{
	static double Sigmoid(double _x)
	{
		return 1 / (1 + exp(-_x));
	}

	static double Sigmoid_Derivative(double x)
	{
		double y = Sigmoid(x);
		return y * (1 - y);
	}

	static double Step(double _x)
	{
		return _x > 0 ? 1 : 0;
	}

	static double ReLU(double _x)
	{
		return _x > 0 ? _x : 0;
	}

	static double ReLU_Derivative(double _x)
	{
		double y = ReLU(_x);
		return 0;
	}
};