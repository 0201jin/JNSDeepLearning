#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include <cuda_runtime.h>

#include "../Activation_Function.h"
#include "device_launch_parameters.h"

using namespace Activation_Function;

class RNN_Layer
{
public:
	RNN_Layer();

	double Calculate_M2O(const vector<double> _InputData);
	void Train_M2O(const vector<double> _InputData, const double _Answer);

protected:
	double m_dXWeight, m_dHWeight, m_dYWeight;
	double m_dHBias, m_dYBias;
	double m_dLastH;

	vector<double> m_vH;
};

class RNN_Network
{
public:
	RNN_Network();

	double Calculate_M2O(const vector<double> _InputData);
	void Train_M2O(const vector<vector<double>> _InputData, const vector<double> _Answer);

private:
	RNN_Layer m_Layer;
};