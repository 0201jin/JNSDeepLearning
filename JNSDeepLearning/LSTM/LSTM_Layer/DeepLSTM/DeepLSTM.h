#pragma once
#include "../LSTM_Layer.h"

class DeepLSTM
{
public:
  DeepLSTM(int _NeuronSize);
  
  double Calculate_M2O(vector<double> _InputData);
  void Train_M2O(vector<vector<double>> _InputData, vector<double> _TrainData);
  
private:
  vector<LSTM_Neuron> m_vNeuron;
};
