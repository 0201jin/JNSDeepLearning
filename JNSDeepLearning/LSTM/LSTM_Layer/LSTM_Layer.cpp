#include "LSTM_Layer.h"

#define LEARN_RATE 0.0025

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
	vector<double> Y = Calculate_Y(_InputData);

	return Y[Y.size() - 1];
}

void LSTM_Neuron::Train_M2O(vector<double> _InputData, double _TrainData)
{
	Train_Y(_InputData, _TrainData);
}

vector<double> LSTM_Neuron::Calculate_H(vector<double> _InputData)
{
	ClearLayer();

	m_vLastInput = _InputData;
	vector<double> H;

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

		Mem_CH.push_back(ch);
		Mem_Gate.push_back(gate);
		H.push_back(ch.H);
	}

	return H;
}

vector<double> LSTM_Neuron::Calculate_Y(vector<double> _InputData)
{
	ClearLayer();

	m_vLastInput = _InputData;

	for (int i = 0; i < _InputData.size(); ++i)
	{
		Gate gate;
		//Mem_CH가 0으로 변하면서 Mem_Gate도 0으로 변함.
		//m_XWeight, m_HWeight, m_HBias를 출력하면서 값 확인하기.
		//Weight와 Bias 모두 특정 값으로 고정
		
		//7.8 gate와 Input값 모두 확인해보기
		gate.f = Sigmoid(m_XWeight.f * _InputData[i] + m_HWeight.f * Mem_CH[i].H + m_HBias.f);
		gate.i = Sigmoid(m_XWeight.i * _InputData[i] + m_HWeight.i * Mem_CH[i].H + m_HBias.i);
		gate.c_ = Sigmoid(m_XWeight.c_ * _InputData[i] + m_HWeight.c_ * Mem_CH[i].H + m_HBias.c_);
		gate.g = Tanh(m_XWeight.g * _InputData[i] + m_HWeight.g * Mem_CH[i].H + m_HBias.g);

		CH ch;
		ch.C = Mem_CH[i].C * gate.f + gate.i * gate.g;
		ch.H = gate.c_ * Tanh(ch.C);

		double Y = ch.H * m_YWeight + m_YBias;

		cout << "Gate : " << gate << " Input : " << _InputData[i] << endl;

		Mem_CH.push_back(ch);
		Mem_Gate.push_back(gate);
		m_vY.push_back(Y);
	}

	return m_vY;
}

queue<double> LSTM_Neuron::Train_Y(vector<double> _InputData, double _TrainData, double _Learning_Rate)
{
	double Y = Calculate_Y(_InputData)[_InputData.size() - 1];

	queue<double> dX;

	CH prev_dCH;

	double dy = 2 * (Y - _TrainData);
	double dh = dy * m_YWeight;
	double dc = Tanh_Derivative(Mem_CH[Mem_CH.size() - 1].C) * dh * Mem_Gate[Mem_CH.size() - 2].c_ + prev_dCH.C;

	m_YWeight -= dy * Mem_CH[Mem_CH.size() - 1].H * LEARN_RATE;
	m_YBias -= dy * LEARN_RATE;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		Gate gate;
		Gate XWeight = m_XWeight;
		Gate HWeight = m_HWeight;

		dh = dh + prev_dCH.H;
		dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Tanh_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Sigmoid_Derivative(Mem_Gate[i].c_);

		m_XWeight.f -= gate.f * _InputData[i] * _Learning_Rate;
		m_XWeight.i -= gate.i * _InputData[i] * _Learning_Rate;
		m_XWeight.g -= gate.g * _InputData[i] * _Learning_Rate;
		m_XWeight.c_ -= gate.c_ * _InputData[i] * _Learning_Rate;

		m_HWeight.f -= gate.f * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.i -= gate.i * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.g -= gate.g * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.c_ -= gate.c_ * Mem_CH[i].H * _Learning_Rate;

		m_HBias.f -= gate.f * _Learning_Rate;
		m_HBias.i -= gate.i * _Learning_Rate;
		m_HBias.g -= gate.g * _Learning_Rate;
		m_HBias.c_ -= gate.c_ * _Learning_Rate;

		//(Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_) * 
		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (HWeight.f + HWeight.i + HWeight.g + HWeight.c_);
		prev_dCH.C = Mem_Gate[i].f * dc;

		dX.push((gate.f + gate.i + gate.g + gate.c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_));
	}

	return dX;
}

queue<double> LSTM_Neuron::Train_H(vector<double> _InputData, queue<double> _TrainData, double _Learning_Rate)
{
	vector<double> Y = Calculate_H(_InputData);

	queue<double> dX;

	CH prev_dCH;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double TrainData = _TrainData.front();
		_TrainData.pop();

		double dh = TrainData + prev_dCH.H;
		double dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		Gate gate;
		Gate XWeight = m_XWeight;
		Gate HWeight = m_HWeight;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Tanh_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Sigmoid_Derivative(Mem_Gate[i].c_);

		m_XWeight.f -= gate.f * _InputData[i] * _Learning_Rate;
		m_XWeight.i -= gate.i * _InputData[i] * _Learning_Rate;
		m_XWeight.g -= gate.g * _InputData[i] * _Learning_Rate;
		m_XWeight.c_ -= gate.c_ * _InputData[i] * _Learning_Rate;

		m_HWeight.f -= gate.f * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.i -= gate.i * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.g -= gate.g * Mem_CH[i].H * _Learning_Rate;
		m_HWeight.c_ -= gate.c_ * Mem_CH[i].H * _Learning_Rate;

		m_HBias.f -= gate.f * _Learning_Rate;
		m_HBias.i -= gate.i * _Learning_Rate;
		m_HBias.g -= gate.g * _Learning_Rate;
		m_HBias.c_ -= gate.c_ * _Learning_Rate;

		//double threshold = 0.25;

		//m_XWeight.f = Gradient_Clipping(m_XWeight.f - gate.f * _InputData[i] * _Learning_Rate, threshold);
		//m_XWeight.i = Gradient_Clipping(m_XWeight.i - gate.i * _InputData[i] * _Learning_Rate, threshold);
		//m_XWeight.g = Gradient_Clipping(m_XWeight.g - gate.g * _InputData[i] * _Learning_Rate, threshold);
		//m_XWeight.c_ = Gradient_Clipping(m_XWeight.c_ - gate.c_ * _InputData[i] * _Learning_Rate, threshold);

		//m_HWeight.f = Gradient_Clipping(m_HWeight.f - gate.f * Mem_CH[i].H * _Learning_Rate, threshold);
		//m_HWeight.i = Gradient_Clipping(m_HWeight.i - gate.i * Mem_CH[i].H * _Learning_Rate, threshold);
		//m_HWeight.g = Gradient_Clipping(m_HWeight.g - gate.g * Mem_CH[i].H * _Learning_Rate, threshold);
		//m_HWeight.c_ = Gradient_Clipping(m_HWeight.c_ - gate.c_ * Mem_CH[i].H * _Learning_Rate, threshold);

		//m_HBias.f = Gradient_Clipping(m_HBias.f - gate.f * _Learning_Rate, threshold);
		//m_HBias.i = Gradient_Clipping(m_HBias.i - gate.i * _Learning_Rate, threshold);
		//m_HBias.g = Gradient_Clipping(m_HBias.g - gate.g * _Learning_Rate, threshold);
		//m_HBias.c_ = Gradient_Clipping(m_HBias.c_ - gate.c_ * _Learning_Rate, threshold);

		prev_dCH.C = Mem_Gate[i].f * dc;
		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (HWeight.f + HWeight.i + HWeight.g + HWeight.c_);

		dX.push((gate.f + gate.i + gate.g + gate.c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_));
	}

	return dX;
}

queue<double> LSTM_Neuron::Train_Y_Adam(vector<double> _InputData, double _TrainData, double _Learning_Rate, double* _m, double* _v)
{
	double Y = Calculate_Y(_InputData)[_InputData.size() - 1];

	queue<double> dX;

	CH prev_dCH;

	double dy = 2 * (Y - _TrainData);
	double dh = dy * m_YWeight;
	double dc = Tanh_Derivative(Mem_CH[Mem_CH.size() - 1].C) * dh * Mem_Gate[Mem_CH.size() - 2].c_ + prev_dCH.C;

	m_YWeight -= dy * Mem_CH[Mem_CH.size() - 1].H * LEARN_RATE;
	m_YBias -= dy * LEARN_RATE;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		Gate gate;
		Gate XWeight = m_XWeight;
		Gate HWeight = m_HWeight;

		dh = dh + prev_dCH.H;
		dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Tanh_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Sigmoid_Derivative(Mem_Gate[i].c_);

		Adam(&m_XWeight.f, gate.f * _InputData[i], _m, _v);
		Adam(&m_XWeight.i, gate.i * _InputData[i], _m, _v);
		Adam(&m_XWeight.g, gate.g * _InputData[i], _m, _v);
		Adam(&m_XWeight.c_, gate.c_ * _InputData[i], _m, _v);

		Adam(&m_HWeight.f, gate.f * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.i, gate.i * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.g, gate.g * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.c_, gate.c_ * Mem_CH[i].H, _m, _v);

		Adam(&m_HBias.f, gate.f, _m, _v);
		Adam(&m_HBias.i, gate.i, _m, _v);
		Adam(&m_HBias.g, gate.g, _m, _v);
		Adam(&m_HBias.c_, gate.c_, _m, _v);

		//gate가 0으로 나옴

		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (HWeight.f + HWeight.i + HWeight.g + HWeight.c_);
		prev_dCH.C = Mem_Gate[i].f * dc;

		dX.push((gate.f + gate.i + gate.g + gate.c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_));
	}

	return dX;
}

queue<double> LSTM_Neuron::Train_H_Adam(vector<double> _InputData, queue<double> _TrainData, double _Learning_Rate, double* _m, double* _v)
{
	vector<double> Y = Calculate_H(_InputData);

	queue<double> dX;

	CH prev_dCH;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double TrainData = _TrainData.front();
		_TrainData.pop();

		double dh = TrainData + prev_dCH.H;
		double dc = Tanh_Derivative(Mem_CH[i + 1].C) * dh * Mem_Gate[i].c_ + prev_dCH.C;

		Gate gate;
		Gate XWeight = m_XWeight;
		Gate HWeight = m_HWeight;

		gate.f = Mem_CH[i].C * dc * Sigmoid_Derivative(Mem_Gate[i].f);
		gate.i = Mem_Gate[i].g * dc * Sigmoid_Derivative(Mem_Gate[i].i);
		gate.g = Mem_Gate[i].i * dc * Tanh_Derivative(Mem_Gate[i].g);
		gate.c_ = Tanh(Mem_CH[i + 1].C) * dh * Sigmoid_Derivative(Mem_Gate[i].c_);

		Adam(&m_XWeight.f, gate.f * _InputData[i], _m, _v);
		Adam(&m_XWeight.i, gate.i * _InputData[i], _m, _v);
		Adam(&m_XWeight.g, gate.g * _InputData[i], _m, _v);
		Adam(&m_XWeight.c_, gate.c_ * _InputData[i], _m, _v);

		Adam(&m_HWeight.f, gate.f * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.i, gate.i * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.g, gate.g * Mem_CH[i].H, _m, _v);
		Adam(&m_HWeight.c_, gate.c_ * Mem_CH[i].H, _m, _v);

		Adam(&m_HBias.f, gate.f, _m, _v);
		Adam(&m_HBias.i, gate.i, _m, _v);
		Adam(&m_HBias.g, gate.g, _m, _v);
		Adam(&m_HBias.c_, gate.c_, _m, _v);

		prev_dCH.C = Mem_Gate[i].f * dc;
		prev_dCH.H = (gate.f + gate.i + gate.g + gate.c_) * (HWeight.f + HWeight.i + HWeight.g + HWeight.c_);

		dX.push((gate.f + gate.i + gate.g + gate.c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_));
	}

	return dX;
}
