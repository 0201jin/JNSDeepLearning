#include "DeepLSTM.h"

DeepLSTM::DeepLSTM(int _NeuronSize)
{
  for(int i = 0; i < _NeuronSize; ++i)
  {
    m_vNeuron.push_back(LSTM_Neuron);
  }
}

double DeepLSTM::Calculate_M2O(vector<double> _InputData)
{
  vector<double> Cal = _InputData;
  
  for(vector<LSTM_Neuron>::iterator iter = m_vNeuron.begin(); iter != m_vNeuron.end() - 1; ++iter)
  {
    
  }
}

void DeepLSTM::Train_M2O(vector<double> _InputData, double _TrainData)
{
}
