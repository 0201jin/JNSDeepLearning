#include "Single_Layer_Perceptron.h"

Single_Neuron::Single_Neuron()
{
}

Single_Neuron::Single_Neuron(size_t _input_size)
{
	m_vWeight = (double*)malloc(sizeof(double) * _input_size);

	m_input_size = _input_size;

	Reset();
}

void Single_Neuron::Reset()
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

double Single_Neuron::Calculate(const vector<double>& _x)
{
	double wx = 0.0;
	for (size_t i = 0; i < m_input_size; ++i)
	{
		wx += m_vWeight[i] * _x[i];
	}

	return ReLU(wx + m_dBias);
}

void Single_Neuron::Train(int _train_num, double _a, vector<pair<vector<double>, double>> _train_data)
{
	size_t input_size = _train_data[0].first.size();

	for (int j = 0; j < _train_num; j++)
	{
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
}

void Single_Neuron::Test()
{

}
