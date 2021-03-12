#pragma once
#include <iostream>

namespace Activation_Function
{
	static double Sigmoid(double _x)
	{
		return 1 / (1 + exp(-_x));
	}

	static double Step(double _x)
	{
		return _x > 0 ? 1 : 0;
	}

	static double ReLU(double _x)
	{
		return _x > 0 ? _x : 0;
	}
};