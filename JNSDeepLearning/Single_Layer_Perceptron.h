#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>

#include "Activation_Function.h"

using namespace Activation_Function;
using namespace std;

class Neuron
{
public:
	Neuron();
	Neuron(size_t _input_size);
	double Calculate(const std::vector<double>& _x);
	void Train(double _a, const vector<pair<vector<double>, double>>& _train_data);

private:
	void Reset();

private:
	vector<double> m_vWeight;
	double m_dBias;
};
