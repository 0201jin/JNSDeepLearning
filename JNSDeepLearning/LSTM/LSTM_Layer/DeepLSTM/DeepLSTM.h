#pragma once
#include "../LSTM_Layer.h"

class DeepLSTM
{
public:
	DeepLSTM(int _NeuronSize);

	double Calculate_M2O(vector<double> _InputData);
	void Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData);

	vector<double> Calculate_M2M(vector<double> _InputData);
	void Train_M2M(vector<vector<double>> _InputData, vector<vector<double>> _TrainData);

	void PrintWeight();

private:
	vector<LSTM_Neuron> m_vNeuron;
	LSTM_Neuron m_OutputNeuron;
	vector<double> m_m;
	vector<double> m_v;
	double om, ov;
};
