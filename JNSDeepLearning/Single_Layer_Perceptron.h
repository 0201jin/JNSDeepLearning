#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include <cuda_runtime.h>

#include "Activation_Function.h"
#include "device_launch_parameters.h"

using namespace Activation_Function;
using namespace std;


class Neuron
{
public:
	Neuron();
	Neuron(size_t _input_size);
	double Calculate(const std::vector<double>& _x);
	//void Train(int _train_num, double _a, const vector<pair<double*, double>> _train_data);
	void Train(int _train_num, double _a, vector<pair<vector<double>, double>> _train_data);
	void Test();

private:
	void Reset();

private:
	double* m_vWeight;
	double m_dBias;
	size_t m_input_size;
};
