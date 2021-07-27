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
	vector<double> Y = Calculate_H(_InputData);

	return Y;
}

vector<double> LSTM_Neuron::Train_M2M(vector<double> _InputData, vector<double> _TrainData)
{
	static double m, v = 0;

	queue<double> aa;
	for (int i = 0; i < _TrainData.size(); ++i)
		aa.push(_TrainData[i]);

	aa = Queue_Reverse_Function(aa);
	//값을 거꾸로 넣어야함.
	Train_H_Adam(_InputData, aa, 0.00025, &m, &v);

	return vector<double>();
}

double LSTM_Neuron::Calculate_M2O(vector<double> _InputData)
{
	vector<double> Y = Calculate_H(_InputData);
	Y = Calculate_O(Y);

	return Y[Y.size() - 1];
}

void LSTM_Neuron::Train_M2O(vector<double> _InputData, double _TrainData)
{
	static double m, v = 0;
	Train_Y_Adam(_InputData, _TrainData, 0.00025, &m, &v);
}

vector<double> LSTM_Neuron::Calculate_H(vector<double> _InputData)
{
	ClearLayer();

	m_vLastInput = _InputData;

	for (int i = 0; i < _InputData.size(); ++i)
	{
		//Weight와 Bias가 정상적으로 조정이 안됨
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
		m_vY.push_back(ch.H);
	}

	return m_vY;
}

vector<double> LSTM_Neuron::Calculate_O(vector<double> _InputData)
{
	ClearLayer();
	
	m_vLastInput = _InputData;

	for (vector<double>::iterator iter = _InputData.begin(); iter != _InputData.end(); ++iter)
	{
		double Y = (*iter) * m_YWeight + m_YBias;

		m_vY.push_back(Y);
	}

	return m_vY;
}

queue<double> LSTM_Neuron::Train_Y_Adam(vector<double> _InputData, double _TrainData, double _Learning_Rate, double* _m, double* _v)
{
	double Y = Calculate_H(_InputData)[_InputData.size() - 1];

	queue<double> dX;

	CH prev_dCH;

	double dy = 2 * (Y - _TrainData);
	double dh = dy;
	double dc = Tanh_Derivative(Mem_CH[Mem_CH.size() - 1].C) * dh * Mem_Gate[Mem_CH.size() - 2].c_ + prev_dCH.C;

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

	CH prev_dCH = Mem_CH[Mem_CH.size() - 1];
	
	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double TrainData = _TrainData.front();
		_TrainData.pop();

		double dy = 2 * (Y[i] - TrainData);
		double dh = dy + prev_dCH.H;
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
		
		double ddx = _InputData[i] - ((Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_)) * dy * 0.0025;
		dX.push((Mem_Gate[i].f + Mem_Gate[i].i + Mem_Gate[i].g + Mem_Gate[i].c_) * (XWeight.f + XWeight.i + XWeight.g + XWeight.c_));
	}

	return dX;
}

queue<double> LSTM_Neuron::Train_O_Adam(vector<double> _InputData, queue<double> _TrainData, double* _m, double* _v)
{
	vector<double> Y = Calculate_O(_InputData);

	queue<double> dX;
	
	double SigY = 0;
	double SigTD = 0;
	double SigX = 0;

	for (int i = _InputData.size() - 1; i >= 0; --i)
	{
		double YW = m_YWeight;
		double YB = m_YBias;
		
		double TrainData = _TrainData.front();
		_TrainData.pop();

		SigY += Y[i];
		SigTD += TrainData;
		SigX += _InputData[i];
		
		double dy = 2 * (Y[i] - TrainData);

		/*Adam(&m_YWeight, _InputData[i], _m, _v);
		Adam(&m_YBias, 1, _m, _v);*/

		YW -= _InputData[i] * dy * 0.0025;
		YB -= dy * 0.0025;

		//dx를 만드는 코드

		dX.push(YW * dy);
	}
	
	m_YWeight -= SigX * (2 * (SigY - SigTD)) * 0.0025;
	m_YBias -= (2 * (SigY - SigTD)) * 0.0025;

	return dX;
}
