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

	m_dXWeight = dist(random);
	m_dHWeight = dist(random);
	m_dYWeight = dist(random);
	m_dHBias = 0;
	m_dYBias = 0;
}

void RNN_Layer::printY()
{
	for (vector<double>::iterator iter = m_vY.begin(); iter != m_vY.end(); ++iter)
	{
		cout << *iter << endl;
	}
}

double RNN_Layer::Calculate_M2O(const vector<double> _InputData)
{
	m_vH.clear();
	m_vY.clear();

	double H = 0;

	m_vH.push_back(0);
	
	for (vector<double>::const_iterator iter = _InputData.begin(); iter != _InputData.end(); ++iter)
	{
		H = Tanh(m_dHWeight * H + m_dXWeight * (*iter) + m_dHBias);

		m_vH.push_back(H);
		m_vY.push_back(H * m_dYWeight + m_dYBias);
	}

	m_dLastH = H;

	return H * m_dYWeight + m_dYBias;
}

void RNN_Layer::Train_M2O(const vector<double> _InputData, const double _Answer)
{
	//https://blog.naver.com/sooftware/221750172371
	//https://blog.naver.com/staystays/222279290417
	//https://blog.naver.com/infoefficien/221209484540
	
	double rate = 0.001;

	double Y = Calculate_M2O(_InputData);

	double LastBias = m_dYBias;
	double LastWeight = m_dYWeight;

	double dE = 2 * (Y - _Answer);
	double dTanh = Tanh_Derivative(Y);
	double dH = dE * LastWeight * dTanh;

	m_dYWeight -= (m_vH[m_vH.size() - 1] * dE) * rate;
	m_dYBias -= dE * rate;
	
	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double Wh = m_dHWeight;

		m_dXWeight -= (dH * _InputData[i]) * rate;
		m_dHWeight -= (dH * m_vH[i]) * rate;
		m_dHBias -= (dH) * rate;

		dH *= Wh;
	}
}
