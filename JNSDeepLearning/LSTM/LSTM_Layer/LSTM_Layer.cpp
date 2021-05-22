#include "LSTM_Layer.h"

LSTM_Layer::LSTM_Layer()
{
	m_dXWeight.Init(); 
	m_dHWeight.Init();

	ClearLayer();
}

void LSTM_Layer::ClearLayer()
{
	Mem_Gate.clear();
	Mem_CH.clear();
	m_vY.clear();

	Mem_Gate.push_back(Gate());
	Mem_CH.push_back(pair<double, double>(0, 0));
}

vector<double> LSTM_Layer::Calculate_M2M(vector<double> _InputData)
{
	//wh * h + wx * x + b

	for (int i = 0; i < _InputData.size(); ++i)
	{
		Gate gate;

		gate.f = Sigmoid(m_dHWeight.f * Mem_CH[i].second + m_dXWeight.f * _InputData[i] + m_dBias.f);
		gate.i = Sigmoid(m_dHWeight.i * Mem_CH[i].second + m_dXWeight.i * _InputData[i] + m_dBias.i);
		gate.o = Sigmoid(m_dHWeight.o * Mem_CH[i].second + m_dXWeight.o * _InputData[i] + m_dBias.o);
		gate.c = Tanh(m_dHWeight.c * Mem_CH[i].second + m_dXWeight.c * _InputData[i] + m_dBias.c);

		double C = Mem_CH[i].first * gate.f + (gate.c * gate.i);
		double H = gate.o * Tanh(C);

		Mem_Gate.push_back(gate);
		Mem_CH.push_back(pair<double, double>(C, H));

		m_vY.push_back(H);
	}

	return m_vY;
}

void LSTM_Layer::Train_M2M(vector<double> _InputData, vector<double> _TrainData)
{
	vector<double> Y = Calculate_M2M(_InputData);

	for (int i = 0; i < _InputData.size(); ++i)
	{
		double dy = 2 * (Y[i] - _TrainData[i]);
		double dh = dy;
	}
}


LSTM_Network::LSTM_Network()
{
}

vector<double> LSTM_Network::Calculate_M2M(vector<double> _InputData)
{
	return m_Layer.Calculate_M2M(_InputData);
}

void LSTM_Network::Train_M2M(vector<vector<double>> _InputData, vector<vector<double>> _TrainData)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		m_Layer.Train_M2M(_InputData[i], _TrainData[i]);
	}
}
