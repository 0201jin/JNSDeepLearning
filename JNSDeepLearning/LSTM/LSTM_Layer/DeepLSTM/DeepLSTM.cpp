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
	vector<double> Cal = _InputData;

	for (vector<LSTM_Neuron>::iterator iter = m_vNeuron.begin(); iter != m_vNeuron.end() - 1; ++iter)
	{
		Cal = (*iter).Calculate_H(Cal);
	}

	vector<double> Y = m_vNeuron[m_vNeuron.size() - 1].Calculate_M2M(Cal);

	return Y[Y.size() - 1];
}

void DeepLSTM::Train_M2O(vector<double> _InputData, double _TrainData)
{
	double Y = Calculate_M2O(_InputData);
	
	vector<double> dh;
	
	dh = m_vNeuron[m_vNeuron.size() - 1].Train_M2O(m_vNeuron[m_vNeuron.size() - 1].GetLastInput(), _TrainData);

	for (int i = m_vNeuron.size() - 2; i >= 0; --i)
	{
		dh = m_vNeuron[i].Train_H(m_vNeuron[i].GetLastInput(), dh);
	}
}
