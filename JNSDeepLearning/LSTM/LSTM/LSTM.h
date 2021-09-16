#pragma once
#include <Activation_Function.h>

using namespace Activation_Function;

class LSTM_Neuron
{
public:
	LSTM_Neuron();
	~LSTM_Neuron();

protected:
	double m_dXWeight, m_dHWeight, m_dYWeight;
	double m_dHBias, m_dYBias;

	vector<double> m_vH;
	vector<double> m_vY;
};

class LSTM_Network
{
public:
	LSTM_Network();
	~LSTM_Network();

protected:
	LSTM_Neuron m_Neuron;
};