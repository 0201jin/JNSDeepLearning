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
	double f = 0;
	double i = 0;
	double g = 0;
	double o = 0;
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

		m_vXGateWeight.f = dist(random);
		m_vXGateWeight.i = dist(random);
		m_vXGateWeight.g = dist(random);
		m_vXGateWeight.o = dist(random);

		m_vHGateWeight.f = dist(random);
		m_vHGateWeight.i = dist(random);
		m_vHGateWeight.g = dist(random);
		m_vHGateWeight.o = dist(random);

		m_dYWeight = dist(random);

		m_vH.push_back(0);
		m_vC.push_back(0);
	}
	~LSTM_Neuron() {}

	void Calculate(const vector<T> _InputData, T& _Answer)
	{
		m_vGate.clear();
		m_vC.clear();
		m_vH.clear();
		m_vH.push_back(0);
		m_vC.push_back(0);

		for (int Index = 0; Index < _InputData.size(); ++Index)
		{
			double f = Sigmoid((m_vXGateWeight.f * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.f) + m_vGateBias.f);
			double i = Sigmoid((m_vXGateWeight.i * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.i) + m_vGateBias.i);
			double o = Sigmoid((m_vXGateWeight.o * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.o) + m_vGateBias.o);
			double g = tanh((m_vXGateWeight.g * _InputData[Index]) + (m_vH[Index] * m_vHGateWeight.g) + m_vGateBias.g);

			double c = f * m_vC[Index] + i * g;
			double h = o * tanh(c);

			m_vGate.push_back(LSTM_Gate({ f, i, g, o }));
			m_vC.push_back(c);
			m_vH.push_back(h);
		}

		_Answer = m_vH[_InputData.size()] * m_dYWeight + m_dYBias;
		//cout << _Answer << endl;
	}

	void Train(const vector<T> _InputData, const T _Answer)
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

			double ddc = Tanh_Derivative(m_vC[Index + 1]) * m_vGate[Index].o * dh;

			double ddo = tanh(m_vGate[Index].g) * dh * Sigmoid_Derivative(m_vGate[Index].o);
			double ddg = m_vGate[Index].i * ddc * Tanh_Derivative(m_vGate[Index].g);
			double ddi = m_vGate[Index].g * ddc * Sigmoid_Derivative(m_vGate[Index].i);
			double ddf = m_vC[Index + 1] * ddc * Sigmoid_Derivative(m_vGate[Index].f);

			m_vGateBias.f -= ddf * LEARN_RATE;
			m_vGateBias.i -= ddi * LEARN_RATE;
			m_vGateBias.g -= ddg * LEARN_RATE;
			m_vGateBias.o -= ddo * LEARN_RATE;

			m_vHGateWeight.f -= m_vH[Index] * ddf * LEARN_RATE;
			m_vHGateWeight.i -= m_vH[Index] * ddi * LEARN_RATE;
			m_vHGateWeight.g -= m_vH[Index] * ddg * LEARN_RATE;
			m_vHGateWeight.o -= m_vH[Index] * ddo * LEARN_RATE;

			m_vXGateWeight.f -= _InputData[Index] * ddf * LEARN_RATE;
			m_vXGateWeight.i -= _InputData[Index] * ddi * LEARN_RATE;
			m_vXGateWeight.g -= _InputData[Index] * ddg * LEARN_RATE;
			m_vXGateWeight.o -= _InputData[Index] * ddo * LEARN_RATE;
		}
	}

	//vector<T>로 하나 만들기

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

template <typename T>
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

protected:
	LSTM_Neuron<T> m_Neuron;
};