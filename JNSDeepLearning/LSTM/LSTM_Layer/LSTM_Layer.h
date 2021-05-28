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
	double g = -1;
	double c_ = -1;

	void Init()
	{
		random_device rd;
		mt19937 random(rd());
		uniform_real_distribution<double> dist(-1, 1);

		f = dist(random);
		i = dist(random);
		g = dist(random);
		c_ = dist(random);
	}

	double  plusAll()
	{
		return f + i + g + c_;
	}
};

struct CH
{
	double C = 0;
	double H = 0;
};

class LSTM_Neuron
{
public:
	LSTM_Neuron();

	void ClearLayer();

	void printY()
	{
		for (vector<double>::iterator iter = m_vY.begin(); iter != m_vY.end(); ++iter)
		{
			cout << *iter << endl;
		}
	}

	vector<double> Calculate_M2M(vector<double> _InputData);
	vector<double> Train_M2M(vector<double> _InputData, vector<double> _TrainData);

	double Calculate_M2O(vector<double> _InputData);
	vector<double> Train_M2O(vector<double> _InputData, double _TrainData);

	vector<double> Calculate_H(vector<double> _InputData);
	vector<double> Train_H(vector<double> _InputData, vector<double> _TrainData);
	
	vector<double> GetCH() { return m_vLastH; }

private:
	Gate m_XWeight;
	Gate m_HWeight;
	Gate m_HBias;

	double m_YWeight;
	double m_YBias;

	vector<CH> Mem_CH;
	vector<Gate> Mem_Gate;
	vector<double> m_vY;
	vector<double> m_vLastH;
};

class LSTM_Layer
{
public:
	LSTM_Layer();

	void printY()
	{
		neuron.printY();
	}

	vector<double> Calculate_M2M(vector<double> _InputData);
	void Train_M2M(vector<double> _InputData, vector<double> _TrainData);

	double Calculate_M2O(vector<double> _InputData);
	void Train_M2O(vector<double> _InputData, double _TrainData);

private:
	LSTM_Neuron neuron;
};

class LSTM_Network
{
public:
	LSTM_Network();

	void printY()
	{
		m_Layer.printY();
	}

	vector<double> Calculate_M2M(vector<double> _InputData);
	void Train_M2M(vector<vector<double>> _InputData, vector<vector<double>> _TrainData);

	double Calculate_M2O(vector<double> _InputData);
	void Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData);

private:
	LSTM_Layer m_Layer;
};
