#pragma once
#include <iostream>
#include <queue>
#include <deque>

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

	static double Gradient_Clipping(double _g, double _threshold)
	{
		if (_g >= _threshold)
			return _threshold / _g * _g;
		
		return _g;
	}
	
	static vector<double> Softmax(vector<double> A)
	{
		double C = 0;
		
		for(int i = 0; i < A.size(); ++i)
			if(C < A[i])
				C = A[i];
		
		double sum = 0;
		for(int i = 0; i < A.size(); ++i)
			sum += exp(A[i] - C);
		
		double constant = C + log(sum);
		
		vector<double> y;
		for(int i = 0; i < A.size(); ++i)
			y.push_back(A[i] - constant);
		
		return y;
	}
	
	static vector<double> Softmax_Derivative(vector<double> y, vector<double> t)
	{
		vector<double> dy;
		
		for(int i = 0; i < y.size(); ++i)
			dy.push_back(y[i] - t[i]);
		
		return dy;
	}
};

namespace Optimize_Function
{
	static void Adam(double* _g, double _dg, double* _m, double* _v)
	{
		(*_m) = 0.9 * (*_m) + (1 - 0.9) * _dg;
		(*_v) = 0.999 * (*_v) + (1 - 0.999) * pow(_dg, 2);
		
		float m_ = (*_m) / (1 - 0.9);
		float v_ = (*_v) / (1 - 0.999);
		
		(*_g) -= 0.025 / sqrt(v_+0.00000001) * m_;
	}
};

namespace Action_Function
{
	template<typename T>
	static queue<T> Queue_Reverse_Function(queue<T> _queue)
	{	
		queue<T> Q;
		vector<T> V;

		for(int i = _queue.size() - 1; i >= 0; --i)
		{
			V.push_back(_queue.front());
			_queue.pop();
		}

		for (int i = V.size() - 1; i >= 0; --i)
			Q.push(V[i]);
		
		return Q;
	}
};
