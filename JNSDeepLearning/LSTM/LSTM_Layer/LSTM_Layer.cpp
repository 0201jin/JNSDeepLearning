#include "LSTM_Layer.h"

LSTM_Layer::LSTM_Layer()
{
	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	for (int i = 0; i < 4; ++i)
	{
		m_dXWeight[i] = dist(random);
		m_dHWeight[i] = dist(random);
		m_dBias[i] = -1;
	}

	ClearLayer();
}

void LSTM_Layer::ClearLayer()
{
	Mem_Gate.clear();
	Mem_CH.clear();

	Mem_Gate.push_back(vector<double>({ 0, 0, 0, 0 }));
	Mem_CH.push_back(pair<double, double>(0, 0));
}


LSTM_Network::LSTM_Network()
{
}