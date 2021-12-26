#pragma once
#include "../../Activation_Function.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include "../../CUDA/CUDA_Matrix.h"

#define LEARN_RATE 0.01

using namespace Activation_Function;
using namespace Optimize_Function;

struct LSTM_Gate
{
	double Gate[4] = { 0, 0, 0, 0 }; //f i g o
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

	//vector<T>ˇÎ ÇĎłŞ ¸¸ľéąâ
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

	//M2O
	void Calculate(const vector<T> _InputData, T& _Answer)
	{
		m_Neuron.Calculate(_InputData, _Answer);
	}

	void Train(const vector<T> _InputData, const vector<T> _Answer, int _Sequence)
	{
		for (int i = 0; i < _InputData.size() - (_Sequence-1); ++i)
		{
			int Index = i + _Sequence - 1;
			vector<T> TrainData(_Sequence);
			copy(_InputData.begin() + i, _InputData.begin() + Index + 1, TrainData.begin());
			m_Neuron.Train(TrainData, _Answer[Index]);
			//cout << TrainData[0] << " " << TrainData[4] << endl;
		}
	}

	//M2M
	void Calculate(const vector<T> _InputData, vector<T>& _Answer)
	{
		m_Neuron.Calculate(_InputData, _Answer);
	}

	void TrainM2M(const vector<T> _InputData, const vector<T> _Answer, int _Sequence)
	{
		for (int i = 0; i < _InputData.size() - (_Sequence-1); ++i)
		{
			int Index = i + _Sequence - 1;
			vector<T> TrainData(_Sequence);
			vector<T> AnswerData(_Sequence);
			
			copy(_InputData.begin() + i, _InputData.begin() + Index + 1, TrainData.begin());
			copy(_Answer.begin() + i, _Answer.begin() + Index + 1, AnswerData.begin());
			
			m_Neuron.Train(TrainData, AnswerData);
		}
	}

protected:
	LSTM_Neuron<T> m_Neuron;
};

//M2O
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

	for (int Index = _InputData.size() - 1; Index >= 0; --Index)
	{
		dh += prev_dh;

		double ddc = Tanh_Derivative(m_vC[Index + 1]) * m_vGate[Index].Gate[3] * dh;

		LSTM_Gate Gate;
		Gate.Gate[3] = tanh(m_vGate[Index].Gate[2]) * dh * Sigmoid_Derivative(m_vGate[Index].Gate[3]);	//o
		Gate.Gate[2] = m_vGate[Index].Gate[1] * ddc * Tanh_Derivative(m_vGate[Index].Gate[2]);			//g
		Gate.Gate[1] = m_vGate[Index].Gate[2] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[1]);		//i
		Gate.Gate[0] = m_vC[Index + 1] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[0]);				//f

		m_vGateBias.Gate[0] -= Gate.Gate[0] * LEARN_RATE;
		m_vGateBias.Gate[1] -= Gate.Gate[1] * LEARN_RATE;
		m_vGateBias.Gate[2] -= Gate.Gate[2] * LEARN_RATE;
		m_vGateBias.Gate[3] -= Gate.Gate[3] * LEARN_RATE;

		m_vHGateWeight.Gate[0] -= m_vH[Index] * Gate.Gate[0] * LEARN_RATE;
		m_vHGateWeight.Gate[1] -= m_vH[Index] * Gate.Gate[1] * LEARN_RATE;
		m_vHGateWeight.Gate[2] -= m_vH[Index] * Gate.Gate[2] * LEARN_RATE;
		m_vHGateWeight.Gate[3] -= m_vH[Index] * Gate.Gate[3] * LEARN_RATE;

		m_vXGateWeight.Gate[0] -= _InputData[Index] * Gate.Gate[0] * LEARN_RATE;
		m_vXGateWeight.Gate[1] -= _InputData[Index] * Gate.Gate[1] * LEARN_RATE;
		m_vXGateWeight.Gate[2] -= _InputData[Index] * Gate.Gate[2] * LEARN_RATE;
		m_vXGateWeight.Gate[3] -= _InputData[Index] * Gate.Gate[3] * LEARN_RATE;
	}
}

//M2M
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

		LSTM_Gate Gate;
		Gate.Gate[3] = tanh(m_vGate[Index].Gate[2]) * dh * Sigmoid_Derivative(m_vGate[Index].Gate[3]);//o
		Gate.Gate[2] = m_vGate[Index].Gate[1] * ddc * Tanh_Derivative(m_vGate[Index].Gate[2]);		  //g
		Gate.Gate[1] = m_vGate[Index].Gate[2] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[1]);	  //i
		Gate.Gate[0] = m_vC[Index + 1] * ddc * Sigmoid_Derivative(m_vGate[Index].Gate[0]);			  //f

		m_vGateBias.Gate[0] -= Gate.Gate[0] * LEARN_RATE;
		m_vGateBias.Gate[1] -= Gate.Gate[1] * LEARN_RATE;
		m_vGateBias.Gate[2] -= Gate.Gate[2] * LEARN_RATE;
		m_vGateBias.Gate[3] -= Gate.Gate[3] * LEARN_RATE;

		m_vHGateWeight.Gate[0] -= m_vH[Index] * Gate.Gate[0] * LEARN_RATE;
		m_vHGateWeight.Gate[1] -= m_vH[Index] * Gate.Gate[1] * LEARN_RATE;
		m_vHGateWeight.Gate[2] -= m_vH[Index] * Gate.Gate[2] * LEARN_RATE;
		m_vHGateWeight.Gate[3] -= m_vH[Index] * Gate.Gate[3] * LEARN_RATE;

		m_vXGateWeight.Gate[0] -= _InputData[Index] * Gate.Gate[0] * LEARN_RATE;
		m_vXGateWeight.Gate[1] -= _InputData[Index] * Gate.Gate[1] * LEARN_RATE;
		m_vXGateWeight.Gate[2] -= _InputData[Index] * Gate.Gate[2] * LEARN_RATE;
		m_vXGateWeight.Gate[3] -= _InputData[Index] * Gate.Gate[3] * LEARN_RATE;
	}
}
