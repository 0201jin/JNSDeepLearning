#include "RNN/RNN.h"

RNN_Network::RNN_Network()
{
}

double RNN_Network::Calculate_M2O(const vector<double> _InputData)
{
	return m_Layer.Calculate_M2O(_InputData);
}

double RNN_Network::Train_M2O(const vector<vector<double>> _InputData)
{
	return 0.0;
}

RNN_Layer::RNN_Layer()
{
	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	m_dXWeight = dist(random);
	m_dHWeight = dist(random);
	m_dYWeight = dist(random);
	m_dHBias = -1;
	m_dYBias = -1;
}

double RNN_Layer::Calculate_M2O(const vector<double> _InputData)
{
	double dH = 0;

	for (vector<double>::const_iterator iter = _InputData.begin(); iter != _InputData.end(); ++iter)
	{
		dH = Tanh(m_dHWeight * dH + m_dXWeight * (*iter) + m_dHBias);
	}

	return dH * m_dYWeight + m_dYBias;
}

void RNN_Layer::Train_M2O(const vector<double> _InputData, const double _Answer)
{
	//https://blog.naver.com/sooftware/221750172371
	//https://blog.naver.com/staystays/222279290417
	//https://blog.naver.com/infoefficien/221209484540

	//YWeight, YBias의 값을 수정하는 작업
	double E = pow(_Answer - Calculate_M2O(_InputData), 2);
	m_dYBias += 0;
	m_dYWeight += 0; //dE/dW = 2wx^2 + 2xb - 2xt

	for (vector<double>::const_iterator iter = _InputData.begin(); iter != _InputData.end(); ++iter)
	{

	}
}