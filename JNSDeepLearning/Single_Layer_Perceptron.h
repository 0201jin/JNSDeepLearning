#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
public:
  Neuron(size_t input_size);
  
private:
  void Reset();
  
private:
  vector<double> m_vWeight;
  double m_dBias;
};
