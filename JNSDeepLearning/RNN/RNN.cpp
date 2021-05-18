#include "RNN.h"

#define LEARN_RATE 0.001

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

void RNN_Layer::Clear()
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

	return H * m_dYWeight + m_dYBias;
}

void RNN_Layer::Train_M2O(const vector<double> _InputData, const double _Answer)
{
	//https://blog.naver.com/sooftware/221750172371
	//https://blog.naver.com/staystays/222279290417
	//https://blog.naver.com/infoefficien/221209484540

	double y = Calculate_M2O(_InputData);

	double LastWH = m_dHWeight;

	double dy = 2 * (y - _Answer);
	double dh = dy * m_dYWeight;

	m_dYWeight -= m_vH[m_vH.size() - 1] * dy * LEARN_RATE;
	m_dYBias -= dy * LEARN_RATE;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double dtanh = dh * Tanh_Derivative(m_vH[i + 1]);

		m_dHWeight -= dtanh * m_vH[i] * LEARN_RATE;
		m_dXWeight -= dtanh * _InputData[i] * LEARN_RATE;
		m_dHBias -= dtanh * LEARN_RATE;

		dh = dtanh;
	}
}

vector<double> RNN_Layer::Calculate_O2M(const double _InputData)
{
	m_vH.clear();
	m_vY.clear();

	double H = m_dXWeight * _InputData;

	m_vH.push_back(0);

	for (int i = 0; i < 3; ++i)
	{
		H = Tanh(m_dHWeight * H + m_dHBias);

		m_vH.push_back(H);
		m_vY.push_back(H * m_dYWeight + m_dYBias);
	}

	return m_vY;
}

void RNN_Layer::Train_O2M(const double _InputData, const vector<double> _Answer)
{
	vector<double> y = Calculate_O2M(_InputData);
	
	double LastWy = m_dYWeight;
	
	for(int i = y.size() - 1; i >= 0; --i)
	{
		double dy = 2 (y[i] - _Answer[i]);
		double dh = dy * LasyWy;
		
		m_dYWeight -= m_vH[i] * dy * LEARN_RATE;
		m_dYBias -= dy * LEARN_RATE;
		
		
		
		double dtanh = 0;
	}
}

RNN_Network::RNN_Network()
{
}

void RNN_Network::Clear()
{
	m_Layer.Clear();
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

vector<double> RNN_Network::Calculate_O2M(const double _InputData)
{
	return m_Layer.Calculate_O2M(_InputData);
}

void RNN_Network::Train_O2M(const vector<double> _InputData, const vector<vector<double>> _Answer)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		m_Layer.Train_O2M(_InputData[i], _Answer[i]);
	}
}
