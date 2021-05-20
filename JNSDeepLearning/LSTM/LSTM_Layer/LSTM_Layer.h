#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include <cuda_runtime.h>

#include "../../Activation_Function.h"
#include "device_launch_parameters.h"

using namespace Activation_Function;

class LSTM_Layer
{
public:
	LSTM_Layer();

	void ClearLayer();

private:
	double m_dXWeight[4] = { 0, 0, 0, 0 };
	double m_dHWeight[4] = { 0, 0, 0, 0 };
	double m_dBias[4] = {-1, -1, -1, -1};

	double m_VWeight = 1;
	double m_VBias = 0;

	double c, h;

	vector<pair<double, double>> Mem_CH;
	vector<vector<double>> Mem_Gate;
};

class LSTM_Network
{
public:
	LSTM_Network();

private:
	LSTM_Layer m_Layer;
};
