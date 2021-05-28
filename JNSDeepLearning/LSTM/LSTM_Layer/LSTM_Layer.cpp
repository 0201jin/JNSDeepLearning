#include "LSTM_Layer.h"

#define LEARN_RATE 0.025

LSTM_Layer::LSTM_Layer()
{
}

vector<double> LSTM_Layer::Calculate_M2M(vector<double> _InputData)
{
	//wh * h + wx * x + b

	return neuron.Calculate_M2M(_InputData);
}

void LSTM_Layer::Train_M2M(vector<double> _InputData, vector<double> _TrainData)
{
	neuron.Train_M2M(_InputData, _TrainData);
}

double LSTM_Layer::Calculate_M2O(vector<double> _InputData)
{
	return neuron.Calculate_M2O(_InputData);
}

void LSTM_Layer::Train_M2O(vector<double> _InputData, double _TrainData)
{
	neuron.Train_M2O(_InputData, _TrainData);
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

double LSTM_Network::Calculate_M2O(vector<double> _InputData)
{
	return m_Layer.Calculate_M2O(_InputData);
}

void LSTM_Network::Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		m_Layer.Train_M2O(_InputData[i], _TrainData[i]);
	}
}

LSTM_Neuron::LSTM_Neuron()
{
	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	m_XWeight.Init();
	m_HWeight.Init();
	m_YWeight = dist(random);
	m_YBias = -1;

	ClearLayer();
}

void LSTM_Neuron::ClearLayer()
{
	Mem_Gate.clear();
	Mem_CH.clear();
	m_vY.clear();

	Mem_CH.push_back(CH());
}

vector<double> LSTM_Neuron::Calculate_M2M(vector<double> _InputData)
{
	ClearLayer();

	for (int i = 0; i < _InputData.size(); ++i)
	{
		Gate gate;

		gate.f = Sigmoid(m_XWeight.f * _InputData[i] + m_HWeight.f * Mem_CH[i].H + m_HBias.f);
		gate.i = Sigmoid(m_XWeight.i * _InputData[i] + m_HWeight.i * Mem_CH[i].H + m_HBias.i);
		gate.c_ = Sigmoid(m_XWeight.c_ * _InputData[i] + m_HWeight.c_ * Mem_CH[i].H + m_HBias.c_);
		gate.g = Tanh(m_XWeight.g * _InputData[i] + m_HWeight.g * Mem_CH[i].H + m_HBias.g);

		CH ch;
		ch.C = Mem_CH[i].C * gate.f + gate.i * gate.g;
		ch.H = gate.c_ * Tanh(ch.C);

		double Y = ch.H * m_YWeight + m_YBias;

		m_vLastH.push_back(ch.H);
		Mem_CH.push_back(ch);
		Mem_Gate.push_back(gate);
		m_vY.push_back(Y);
	}

	return m_vY;
}

vector<double> LSTM_Neuron::Train_M2M(vector<double> _InputData, vector<double> _TrainData)
{
	vector<double> Y = Calculate_M2M(_InputData);
	vector<double> dH;
	
	CH prev_dCH;

	for (int i = Y.size() - 1; i >= 0; --i)
	{
		double dy = 2 * (Y[i] - _TrainData[i]);
		double dh = dy * m_YWeight + prev_dCH.H;
		double dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		m_YWeight -= dy * Mem_CH[i + 1].H * LEARN_RATE;
		m_YBias -= dy * LEARN_RATE;

		Gate gate;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Sigmoid_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Tanh_Derivative(Mem_Gate[i].c_);

		m_XWeight.f -= gate.f * _InputData[i] * LEARN_RATE;
		m_XWeight.i -= gate.i * _InputData[i] * LEARN_RATE;
		m_XWeight.g -= gate.g * _InputData[i] * LEARN_RATE;
		m_XWeight.c_ -= gate.c_ * _InputData[i] * LEARN_RATE;

		m_HWeight.f -= gate.f * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.i -= gate.i * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.g -= gate.g * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.c_ -= gate.c_ * Mem_CH[i].H * LEARN_RATE;

		m_HBias.f -= gate.f * LEARN_RATE;
		m_HBias.i -= gate.i * LEARN_RATE;
		m_HBias.g -= gate.g * LEARN_RATE;
		m_HBias.c_ -= gate.c_ * LEARN_RATE;

		prev_dCH.C = Mem_Gate[i].f * dc;
		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_);
		
		dH.push_back(prev_dCH.H);
	}
	
	return dH;
}

double LSTM_Neuron::Calculate_M2O(vector<double> _InputData)
{
	ClearLayer();

	double Y;

	for (int i = 0; i < _InputData.size(); ++i)
	{
		Gate gate;

		gate.f = Sigmoid(m_XWeight.f * _InputData[i] + m_HWeight.f * Mem_CH[i].H + m_HBias.f);
		gate.i = Sigmoid(m_XWeight.i * _InputData[i] + m_HWeight.i * Mem_CH[i].H + m_HBias.i);
		gate.c_ = Sigmoid(m_XWeight.c_ * _InputData[i] + m_HWeight.c_ * Mem_CH[i].H + m_HBias.c_);
		gate.g = Tanh(m_XWeight.g * _InputData[i] + m_HWeight.g * Mem_CH[i].H + m_HBias.g);

		CH ch;
		ch.C = Mem_CH[i].C * gate.f + gate.i * gate.g;
		ch.H = gate.c_ * Tanh(ch.C);

		Y = ch.H * m_YWeight + m_YBias;

		m_vLastH.push_back(ch.H);
		Mem_CH.push_back(ch);
		Mem_Gate.push_back(gate);
	}

	return Y;
}

vector<double> LSTM_Neuron::Train_M2O(vector<double> _InputData, double _TrainData)
{
	double Y = Calculate_M2O(_InputData);
	vector<double> dH;
	
	CH prev_dCH;

	double dy = 2 * (Y - _TrainData);
	double dh = dy * m_YWeight + prev_dCH.H;
	double dc = Tanh_Derivative(Mem_CH[Mem_CH.size() - 1].C) * dh * Mem_Gate[Mem_CH.size() - 2].c_ + prev_dCH.C;

	m_YWeight -= dy * Mem_CH[Mem_CH.size() - 1].H * LEARN_RATE;
	m_YBias -= dy * LEARN_RATE;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		Gate gate;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Sigmoid_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Tanh_Derivative(Mem_Gate[i].c_);

		m_XWeight.f -= gate.f * _InputData[i] * LEARN_RATE;
		m_XWeight.i -= gate.i * _InputData[i] * LEARN_RATE;
		m_XWeight.g -= gate.g * _InputData[i] * LEARN_RATE;
		m_XWeight.c_ -= gate.c_ * _InputData[i] * LEARN_RATE;

		m_HWeight.f -= gate.f * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.i -= gate.i * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.g -= gate.g * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.c_ -= gate.c_ * Mem_CH[i].H * LEARN_RATE;

		m_HBias.f -= gate.f * LEARN_RATE;
		m_HBias.i -= gate.i * LEARN_RATE;
		m_HBias.g -= gate.g * LEARN_RATE;
		m_HBias.c_ -= gate.c_ * LEARN_RATE;
		
		dH.push_back((gate.f + gate.i + gate.g + gate.c_) * (Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_));
	}
	
	return dH;
}

vector<double> LSTM_Neuron::Calculate_H(vector<double> _InputData)
{
	ClearLayer();

	for (int i = 0; i < _InputData.size(); ++i)
	{
		Gate gate;

		gate.f = Sigmoid(m_XWeight.f * _InputData[i] + m_HWeight.f * Mem_CH[i].H + m_HBias.f);
		gate.i = Sigmoid(m_XWeight.i * _InputData[i] + m_HWeight.i * Mem_CH[i].H + m_HBias.i);
		gate.c_ = Sigmoid(m_XWeight.c_ * _InputData[i] + m_HWeight.c_ * Mem_CH[i].H + m_HBias.c_);
		gate.g = Tanh(m_XWeight.g * _InputData[i] + m_HWeight.g * Mem_CH[i].H + m_HBias.g);

		CH ch;
		ch.C = Mem_CH[i].C * gate.f + gate.i * gate.g;
		ch.H = gate.c_ * Tanh(ch.C);

		double Y = ch.H;

		m_vLastH.push_back(ch.H);
		Mem_CH.push_back(ch);
		Mem_Gate.push_back(gate);
		m_vY.push_back(Y);
	}

	return m_vY;
}

vector<double> LSTM_Neuron::Train_H(vector<double> _InputData, vector<double> _TrainData)
{
	CH prev_dCH;
	vector<double> dH;

	for (int i = _TrainData.size() - 1; i >= 0; --i)
	{
		double dh = _TrainData[i] + prev_dCH.H;
		double dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		Gate gate;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Sigmoid_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Tanh_Derivative(Mem_Gate[i].c_);

		m_XWeight.f -= gate.f * _InputData[i] * LEARN_RATE;
		m_XWeight.i -= gate.i * _InputData[i] * LEARN_RATE;
		m_XWeight.g -= gate.g * _InputData[i] * LEARN_RATE;
		m_XWeight.c_ -= gate.c_ * _InputData[i] * LEARN_RATE;

		m_HWeight.f -= gate.f * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.i -= gate.i * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.g -= gate.g * Mem_CH[i].H * LEARN_RATE;
		m_HWeight.c_ -= gate.c_ * Mem_CH[i].H * LEARN_RATE;

		m_HBias.f -= gate.f * LEARN_RATE;
		m_HBias.i -= gate.i * LEARN_RATE;
		m_HBias.g -= gate.g * LEARN_RATE;
		m_HBias.c_ -= gate.c_ * LEARN_RATE;

		prev_dCH.C = Mem_Gate[i].f * dc;
		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_);
		
		dH.push_back(prev_dCH.H);
	}
	
	return dH;
}
