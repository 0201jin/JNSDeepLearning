#include "RNN.h"

RNN_Network::RNN_Network()
{
}

double RNN_Network::Calculate_M2O(const vector<double> _InputData)
{
	return m_Layer.Calculate_M2O(_InputData);
}

void RNN_Network::Train_M2O(const vector<vector<double>> _InputData, const vector<double> _Answer)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		m_Layer.Train_M2O(_InputData[i], _Answer[i]);
	}
}

RNN_Layer::RNN_Layer()
{
	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	m_dXWeight = 0;
	m_dHWeight = 0;
	m_dYWeight = 0;
	m_dHBias = 0;
	m_dYBias = 0;
}

double RNN_Layer::Calculate_M2O(const vector<double> _InputData)
{
	m_vH.clear();

	double dH = 0;

	for (vector<double>::const_iterator iter = _InputData.begin(); iter != _InputData.end(); ++iter)
	{
		dH = Tanh(m_dHWeight * dH + m_dXWeight * (*iter) + m_dHBias);

		m_vH.push_back(dH);
	}

	m_dLastH = dH;

	return dH * m_dYWeight + m_dYBias;
}

void RNN_Layer::Train_M2O(const vector<double> _InputData, const double _Answer)
{
	//https://blog.naver.com/sooftware/221750172371
	//https://blog.naver.com/staystays/222279290417
	//https://blog.naver.com/infoefficien/221209484540
	
	double rate = 0.0025;

	double Y = Calculate_M2O(_InputData);
	double E = pow(_Answer - Y, 2);
	
	double LastBias = m_dYBias;
	double LastWeight = m_dYWeight;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		m_dYWeight -= (2 * _InputData[i] * (Y - _Answer)) * rate;
		m_dYBias -= (2 * (Y - _Answer)) * rate;

		m_dXWeight -= (2 * (Y - _Answer) * LastWeight * Tanh_Derivative(Y) * _InputData[i]) * rate;
		m_dHWeight -= (2 * (Y - _Answer) * LastWeight * Tanh_Derivative(Y) * m_vH[i]) * rate;
		m_dHBias -= (2 * (Y - _Answer) * LastWeight * Tanh_Derivative(Y)) * rate;
	}
}
