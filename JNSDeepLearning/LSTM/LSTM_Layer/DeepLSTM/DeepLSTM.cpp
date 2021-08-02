#include "DeepLSTM.h"

DeepLSTM::DeepLSTM(int _NeuronSize)
{
	for (int i = 0; i < _NeuronSize; ++i)
	{
		m_vNeuron.push_back(LSTM_Neuron());
		m_m.push_back(0);
		m_v.push_back(0);
	}

	m_OutputNeuron = LSTM_Neuron();
}

double DeepLSTM::Calculate_M2O(vector<double> _InputData)
{
	vector<double> Input = _InputData;

	for (int i = 0; i < m_vNeuron.size() - 1; ++i)
	{
		Input = m_vNeuron[i].Calculate_H(Input);
	}

	Input = m_vNeuron[m_vNeuron.size() - 1].Calculate_H(Input);

	Input = m_OutputNeuron.Calculate_O(Input);

	return Input[_InputData.size() - 1];
}

void DeepLSTM::Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		Calculate_M2O(_InputData[i]);

		queue<double> TrainData = m_vNeuron[m_vNeuron.size() - 1].Train_Y_Adam(m_vNeuron[m_vNeuron.size() - 1].GetLastInput(), _TrainData[i], 0.00025, &m_m[m_vNeuron.size() - 1], &m_v[m_vNeuron.size() - 1]);
		TrainData = Queue_Reverse_Function(TrainData);
		for (int j = m_vNeuron.size() - 2; j >= 0; --j)
		{
			TrainData = m_vNeuron[j].Train_H_Adam(m_vNeuron[j].GetLastInput(), TrainData, 0.00025, &m_m[j], &m_v[j]);
		}
	}
}

vector<double> DeepLSTM::Calculate_M2M(vector<double> _InputData)
{
	vector<double> Input = _InputData;

	for (int i = 0; i < m_vNeuron.size(); ++i)
	{
		Input = m_vNeuron[i].Calculate_H(Input);
	}

	//Input = m_OutputNeuron.Calculate_O(Input);

	return Input;
}

void DeepLSTM::Train_M2M(vector<vector<double>> _InputData, vector<vector<double>> _TrainData)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		Calculate_M2M(_InputData[i]);

		queue<double> TrainData;

		for (int j = _TrainData[i].size() - 1; j >= 0; --j)
		{
			TrainData.push(_TrainData[i][j]);
		}

		//TrainData = m_OutputNeuron.Train_O_Adam(m_OutputNeuron.GetLastInput(), TrainData, &om, &ov);
		
		for (int j = m_vNeuron.size() - 1; j >= 0; --j)
		{
			TrainData = m_vNeuron[j].Train_H_Adam(m_vNeuron[j].GetLastInput(), TrainData, 0.025, &m_m[0], &m_v[0]);
		}
	}
}

void DeepLSTM::PrintWeight()
{
	for (int i = 0; i < m_vNeuron.size(); ++i)
	{
		m_vNeuron[i].PrintWeight(i);
	}

	m_OutputNeuron.PrintWeight();
}
