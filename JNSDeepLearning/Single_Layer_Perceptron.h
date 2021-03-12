#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>

using namespace std;

double Sigmoid(double _x)
{
  return 1 / (1 + exp(-_x));
}

double Step(double _x)
{
  return _x > 0 ? 1 : 0;
}

double ReLU(double _x)
{
  return _x > 0 ? x : 0;
}

class Neuron
{
public:
  Neuron(size_t _input_size);
  double Calculate(const std::vector<double>& _x) const;
  void Train(double _a, const vector<pair<vector<double>, double>>& _train_data);
  
private:
  void Reset();
  
private:
  vector<double> m_vWeight;
  double m_dBias;
};
