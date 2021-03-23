#pragma once
#include <iostream>

using namespace std;

namespace Activation_Function
{
	static double Sigmoid(double _x)
	{
		return 1 / (1 + exp(-_x));
	}

	static double Sigmoid_Derivative(double _x)
	{
		double y = Sigmoid(_x);
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
		return _x > 0 ? 1 : 0;
	}

	static double ELU(double _x, double _a = 1)
	{
		return  _x > 0 ? _x : _a * (exp(_x) - 1);
	}

	static double ELU_Derivative(double _x, double _a = 1)
	{
		return _x > 0 ? 1 : ELU(_x, _a) + _a;
	}

	static double Tanh(double _x)
	{
		return 2 * Sigmoid(2 * _x) - 1;
	}

	static double Tanh_Derivative(double _x)
	{
		return 1 - pow(Tanh(_x), 2);
	}
};