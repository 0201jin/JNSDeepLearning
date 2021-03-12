#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>

using namespace std;

class Neuron
{
public:
  Neuron(size_t _input_size);
  double Compute(const std::vector<double>& _x) const;
  void Train(double _a, const vector<pair<vector<double>, double>>& _train_data);
  
private:
  void Reset();
  
private:
  vector<double> m_vWeight;
  double m_dBias;
};
