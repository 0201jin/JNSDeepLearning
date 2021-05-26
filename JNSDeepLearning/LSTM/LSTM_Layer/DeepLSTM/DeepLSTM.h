#pragma once
#include "../LSTM_Layer.h"

class DeepLSTM
{
public:
  double Calculate_M2O(vector<double> _InputData);
  void Train_M2O(vector<double> _InputData, double _TrainData);
  
private:
  vector<LSTM_Neuron> m_vNeuron;
};
