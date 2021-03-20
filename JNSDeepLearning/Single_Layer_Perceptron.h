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


class Single_Neuron
{
public:
	Single_Neuron();
	Single_Neuron(size_t _input_size);
	double Calculate(const vector<double>& _x);
	void Train(int _train_num, double _a, vector<pair<vector<double>, double>> _train_data);
	void CUDA_Train(int _train_num, double _a, vector<pair<vector<double>, double>> _train_data);
	void Test();

private:
	void Reset();

private:
	double* m_vWeight;
	double m_dBias;
	size_t m_input_size;
};