#include "Single_Layer_Perceptron.h"

Neuron::Neuron()
{
}

Neuron::Neuron(size_t _input_size)
{
	m_vWeight = (double*)malloc(sizeof(double) * _input_size);

	m_input_size = _input_size;

	Reset();
}

void Neuron::Reset()
{
	m_dBias = -1;

	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	for (size_t i = 0; i < m_input_size; ++i)
	{
		m_vWeight[i] = dist(random);
	}
}

double Neuron::Calculate(const vector<double>& _x)
{
	if (_x.size() != m_input_size)
		cout << "x.size() != Weight.size()" << endl;

	double wx = 0.0;
	for (size_t i = 0; i < m_input_size; ++i)
	{
		wx += m_vWeight[i] * _x[i];
	}

	return ReLU(wx + m_dBias);
}

void Neuron::Train(int _train_num, double _a, vector<pair<vector<double>, double>> _train_data)
{
	size_t input_size = _train_data[0].first.size();

	if (input_size != m_input_size)
		cout << "input_size != Weights_.size()" << endl;

	for (size_t i = 0; i < _train_data.size(); ++i)
	{
		double o = Calculate(_train_data[i].first);
		double t = _train_data[i].second;

		for (size_t j = 0; j < input_size; ++j)
		{
			m_vWeight[j] += _a * (t - o) * _train_data[i].first[j];
		}

		m_dBias += _a * (t - o);
	}
}

void Neuron::Test()
{

}
