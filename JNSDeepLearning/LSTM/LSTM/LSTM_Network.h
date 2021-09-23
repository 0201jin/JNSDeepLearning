#pragma once
#include <Activation_Function.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>

#define LEARN_RATE 0.01

using namespace Activation_Function;
using namespace Optimize_Function;

struct LSTM_Gate
{
	double Gate[4] = {0, 0, 0, 0}; //f i g o
	/*double f = 0;
	double i = 0;
	double g = 0;
	double o = 0;*/
};

template <typename T>
class LSTM_Neuron
{
public:
	LSTM_Neuron()
	{
		random_device rd;
		mt19937 random(rd());
		uniform_real_distribution<double> dist(-1, 1);

		m_vXGateWeight.Gate[0] = dist(random);
		m_vXGateWeight.Gate[1] = dist(random);
		m_vXGateWeight.Gate[2] = dist(random);
		m_vXGateWeight.Gate[3] = dist(random);

		m_vHGateWeight.Gate[0] = dist(random);
		m_vHGateWeight.Gate[1] = dist(random);
		m_vHGateWeight.Gate[2] = dist(random);
		m_vHGateWeight.Gate[3] = dist(random);

		m_dYWeight = dist(random);

		m_vH.push_back(0);
		m_vC.push_back(0);
	}
	~LSTM_Neuron() {}

	void Calculate(const vector<T> _InputData, T& _Answer);
	void Train(const vector<T> _InputData, const T _Answer);

	//vector<T>로 하나 만들기
	void Calculate(const vector<T> _InputData, vector<T>& _Answer);
	void Train(const vector<T> _InputData, const vector<T> _Answer);

private:
	void Clear()
	{
		m_vGate.clear();
		m_vC.clear();
		m_vH.clear();
		m_vH.push_back(0);
		m_vC.push_back(0);
	}

protected:
	LSTM_Gate m_vXGateWeight;
	LSTM_Gate m_vHGateWeight;
	LSTM_Gate m_vGateBias;

	double m_dYWeight = 0;
	double m_dYBias = 0;

	vector<double> m_vH;
	vector<double> m_vC;
	vector<LSTM_Gate> m_vGate;
	vector<double> m_vY;
};

template <class T>
class LSTM_Network
{
public:
	LSTM_Network() {}
	~LSTM_Network() {}

	void Calculate(const vector<T> _InputData, T& _Answer)
	{
		m_Neuron.Calculate(_InputData, _Answer);
	}

	void Train(const vector<vector<T>> _InputData, const vector<T> _Answer)
	{
		for (int i = 0; i < _InputData.size(); ++i)
		{
			m_Neuron.Train(_InputData[i], _Answer[i]);
		}
	}

	void Calculate(const vector<T> _InputData, vector<T>& _Answer)
	{
		m_Neuron.Calculate(_InputData, _Answer);
	}

	void Train(const vector<vector<T>> _InputData, const vector<vector<T>> _Answer)
	{
		for (int i = 0; i < _InputData.size(); ++i)
		{
			m_Neuron.Train(_InputData[i], _Answer[i]);
		}
	}

protected:
	LSTM_Neuron<T> m_Neuron;
};

template<typename T>
inline void LSTM_Neuron<T>::Calculate(const vector<T> _InputData, T& _Answer)
{
	Clear();

	for (int Index = 0; Index < _InputData.size(); ++Index)
	{
		double f = Sigmoid((m_vXGateWeight.Gate[0] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[0]) + m_vGateBias.Gate[0]);
		double i = Sigmoid((m_vXGateWeight.Gate[1] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[1]) + m_vGateBias.Gate[1]);
		double o = Sigmoid((m_vXGateWeight.Gate[3] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[3]) + m_vGateBias.Gate[3]);
		double g = tanh((m_vXGateWeight.Gate[2] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[2]) + m_vGateBias.Gate[2]);

		double c = f * m_vC[Index] + i * g;
		double h = o * tanh(c);

		m_vGate.push_back(LSTM_Gate({ f, i, g, o }));
		m_vC.push_back(c);
		m_vH.push_back(h);
	}

	_Answer = m_vH[_InputData.size()] * m_dYWeight + m_dYBias;
}

template<typename T>
inline void LSTM_Neuron<T>::Train(const vector<T> _InputData, const T _Answer)
{
	double Y;
	Calculate(_InputData, Y);

	double prev_dh = 0;

	double dY = 2 * (Y - _Answer);
	double dh = dY * m_dYWeight;

	m_dYWeight -= m_vH[m_vH.size() - 1] * dY * LEARN_RATE;
	m_dYBias -= dY * LEARN_RATE;
	//cout << m_dYWeight << endl;
	for (int Index = _InputData.size() - 1; Index >= 0; --Index)
	{
		dh += prev_dh;

		double ddc = Tanh_Derivative(m_vC[Index + 1]) * m_vGate[Index].Gate[3] * dh;

		double ddo = tanh(m_vGate[Index].Gate[2]) * dh * Sigmoid_Derivative(m_vGate[Index].Gate[3]);
		double ddg = m_vGate[Index].Gate[1] * ddc * Tanh_Derivative(m_vGate[Index].Gate[2]);
		double ddi = m_vGate[Index].Gate[2] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[1]);
		double ddf = m_vC[Index + 1] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[0]);

		m_vGateBias.Gate[0] -= ddf * LEARN_RATE;
		m_vGateBias.Gate[1] -= ddi * LEARN_RATE;
		m_vGateBias.Gate[2] -= ddg * LEARN_RATE;
		m_vGateBias.Gate[3] -= ddo * LEARN_RATE;

		m_vHGateWeight.Gate[0] -= m_vH[Index] * ddf * LEARN_RATE;
		m_vHGateWeight.Gate[1] -= m_vH[Index] * ddi * LEARN_RATE;
		m_vHGateWeight.Gate[2] -= m_vH[Index] * ddg * LEARN_RATE;
		m_vHGateWeight.Gate[3] -= m_vH[Index] * ddo * LEARN_RATE;

		m_vXGateWeight.Gate[0] -= _InputData[Index] * ddf * LEARN_RATE;
		m_vXGateWeight.Gate[1] -= _InputData[Index] * ddi * LEARN_RATE;
		m_vXGateWeight.Gate[2] -= _InputData[Index] * ddg * LEARN_RATE;
		m_vXGateWeight.Gate[3] -= _InputData[Index] * ddo * LEARN_RATE;
	}
}

template<typename T>
inline void LSTM_Neuron<T>::Calculate(const vector<T> _InputData, vector<T>& _Answer)
{
	Clear();

	for (int Index = 0; Index < _InputData.size(); ++Index)
	{
		double f = Sigmoid((m_vXGateWeight.Gate[0] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[0]) + m_vGateBias.Gate[0]);
		double i = Sigmoid((m_vXGateWeight.Gate[1] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[1]) + m_vGateBias.Gate[1]);
		double o = Sigmoid((m_vXGateWeight.Gate[3] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[3]) + m_vGateBias.Gate[3]);
		double g = tanh((m_vXGateWeight.Gate[2] * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.Gate[2]) + m_vGateBias.Gate[2]);

		double c = f * m_vC[Index] + i * g;
		double h = o * tanh(c);

		m_vGate.push_back(LSTM_Gate({ f, i, g, o }));
		m_vC.push_back(c);
		m_vH.push_back(h);
		_Answer.push_back(h * m_dYWeight + m_dYBias);
	}
}

template<typename T>
inline void LSTM_Neuron<T>::Train(const vector<T> _InputData, const vector<T> _Answer)
{
	vector<double> Y;
	Calculate(_InputData, Y);

	double prev_dh = 0;

	//cout << m_dYWeight << endl;
	for (int Index = _InputData.size() - 1; Index >= 0; --Index)
	{
		double dY = 2 * (Y[Index] - _Answer[Index]);
		double dh = dY * m_dYWeight;

		m_dYWeight -= m_vH[Index] * dY * LEARN_RATE;
		m_dYBias -= dY * LEARN_RATE;

		dh += prev_dh;

		double ddc = Tanh_Derivative(m_vC[Index + 1]) * m_vGate[Index].Gate[3] * dh;

		double ddo = tanh(m_vGate[Index].Gate[2]) * dh * Sigmoid_Derivative(m_vGate[Index].Gate[3]);
		double ddg = m_vGate[Index].Gate[1] * ddc * Tanh_Derivative(m_vGate[Index].Gate[2]);
		double ddi = m_vGate[Index].Gate[2] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[1]);
		double ddf = m_vC[Index + 1] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[0]);

		m_vGateBias.Gate[0] -= ddf * LEARN_RATE;
		m_vGateBias.Gate[1] -= ddi * LEARN_RATE;
		m_vGateBias.Gate[2] -= ddg * LEARN_RATE;
		m_vGateBias.Gate[3] -= ddo * LEARN_RATE;

		m_vHGateWeight.Gate[0] -= m_vH[Index] * ddf * LEARN_RATE;
		m_vHGateWeight.Gate[1] -= m_vH[Index] * ddi * LEARN_RATE;
		m_vHGateWeight.Gate[2] -= m_vH[Index] * ddg * LEARN_RATE;
		m_vHGateWeight.Gate[3] -= m_vH[Index] * ddo * LEARN_RATE;
					   
		m_vXGateWeight.Gate[0] -= _InputData[Index] * ddf * LEARN_RATE;
		m_vXGateWeight.Gate[1] -= _InputData[Index] * ddi * LEARN_RATE;
		m_vXGateWeight.Gate[2] -= _InputData[Index] * ddg * LEARN_RATE;
		m_vXGateWeight.Gate[3] -= _InputData[Index] * ddo * LEARN_RATE;
	}
}