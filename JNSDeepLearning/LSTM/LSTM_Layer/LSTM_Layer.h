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

	vector<double> Calculate_M2O(double _C, double _H, const vector<double>& _InputData);
	void Train_M2O(double _e, double _a, const vector<vector<double>>& _TrainData);

private:
	double m_dXWeight[4] = { 0, 0, 0, 0 };
	double m_dHWeight[4] = { 0, 0, 0, 0 };
	double m_dBias[4] = {-1, -1, -1, -1};

	double m_VWeight = 1;
	double m_VBias = 0;
};

class LSTM_Network
{
public:
	LSTM_Network();

	vector<double> Calculate_M2O(const vector<double>& _InputData);
	void Train_M2O(const vector<vector<double>>& _TrainData);

private:
	LSTM_Layer m_Layer;
};