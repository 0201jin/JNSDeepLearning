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

class ML_Neuron
{
public:
	ML_Neuron(int _input_size);
	double Calculate_Sigmoid(const std::vector<double>& _x) const;
	double Calculate_ELU(const std::vector<double>& _x) const;
	double Calculate_ReLU(const std::vector<double>& _x) const;

	void Train_Neuron_Sigmoid(double _a, double _e, const vector<double>& _Train_Data);
	void Train_Neuron_ReLU(double _a, double _e, const vector<double>& _Train_Data);
	void Train_Neuron_ELU(double _a, double _e, const vector<double>& _Train_Data);
	void Train_Neuron(double _a, double _e, const vector<double>& _Train_Data);

	int GetInputSize() const
	{
		return m_input_size;
	}

	double GetLastV() const
	{
		return m_LastV;
	}

	double GetLastD() const
	{
		return m_LastD;
	}

	double& GetBias()
	{
		return m_dBias;
	}

	vector<double> GetWeights()
	{
		return m_vWeight;
	}

	const vector<double>& GetLastX() const
	{
		return m_LastX;
	}

private:
	void Reset();

private:
	double m_dBias;
	double m_LastD;
	mutable double m_LastV;

	vector<double> m_vWeight;
	mutable vector<double> m_LastX;

	int m_input_size;
};

class ML_Network
{
public:
	ML_Network(const vector<int>& _layers);
	vector<double> Calculate(const vector<double>& _vx);
	void Train_Network(int _TrainNum, double _a, const vector<pair<vector<double>, vector<double>>>& _Train_Data);

private:
	vector<vector<ML_Neuron>> m_vLayers;
};