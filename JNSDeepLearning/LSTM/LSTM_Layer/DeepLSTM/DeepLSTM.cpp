#include "DeepLSTM.h"

DeepLSTM::DeepLSTM(int _NeuronSize)
{
	for (int i = 0; i < _NeuronSize; ++i)
	{
		m_vNeuron.push_back(LSTM_Neuron());
	}
}

double DeepLSTM::Calculate_M2O(vector<double> _InputData)
{
	vector<double> Input = _InputData;

	for (int i = 0; i < m_vNeuron.size() - 1; ++i)
	{
		Input = m_vNeuron[i].Calculate_H(Input);
	}

	return m_vNeuron[m_vNeuron.size() - 1].Calculate_Y(Input)[_InputData.size() - 1];
}

void DeepLSTM::Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData)
{
	for (int i = 0; i < _InputData.size(); ++i)
	{
		Calculate_M2O(_InputData[i]);

		queue<double> TrainData = m_vNeuron[m_vNeuron.size() - 1].Train_Y(m_vNeuron[m_vNeuron.size() - 1].GetLastInput(), _TrainData[i]);

		for (int j = m_vNeuron.size() - 2; j >= 0; --j)
		{
			TrainData = m_vNeuron[j].Train_H(m_vNeuron[j].GetLastInput(), TrainData);
		}
	}
}

void DeepLSTM::PrintWeight()
{
	for (int i = 0; i < m_vNeuron.size(); ++i)
	{
		m_vNeuron[i].PrintWeight(i);
	}
}
