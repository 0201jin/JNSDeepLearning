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

struct Gate
{
	double f = -1;
	double i = -1;
	double c = -1;
	double o = -1;

	void Init()
	{
		random_device rd;
		mt19937 random(rd());
		uniform_real_distribution<double> dist(-1, 1);

		f = dist(random);
		i = dist(random);
		c = dist(random);
		o = dist(random);
	}
};

class LSTM_Layer
{
public:
	LSTM_Layer();

	void ClearLayer();

	vector<double> Calculate_M2M(vector<double> _InputData);
	void Train_M2M(vector<double> _InputData, vector<double> _TrainData);

private:
	Gate m_dXWeight;
	Gate m_dHWeight;
	Gate m_dBias;

	vector<pair<double, double>> Mem_CH;
	vector<Gate> Mem_Gate;
	vector<double> m_vY;
};

class LSTM_Network
{
public:
	LSTM_Network();

	vector<double> Calculate_M2M(vector<double> _InputData);
	void Train_M2M(vector<vector<double>> _InputData, vector<vector<double>> _TrainData);

private:
	LSTM_Layer m_Layer;
};
