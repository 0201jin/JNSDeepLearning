#include "Multi_Layer_Perceptron.h"

ML_Neuron::ML_Neuron(int _input_size)
{
	m_input_size = _input_size;
	//m_vWeight = (double*)malloc(sizeof(double) * _input_size);
	m_vWeight.resize(_input_size);
	Reset();
}

double ML_Neuron::Calculate(const std::vector<double>& _x) const
{
	double wx = 0.0;

	for (int i = 0; i < m_input_size; ++i)
	{
		wx += m_vWeight[i] * _x[i];
	}

	m_LastV = wx + m_dBias;
	m_LastX = _x;

	return Sigmoid(m_LastV);
}

void ML_Neuron::Train_Neuron(double _a, double _e, const vector<double>& _Train_Data)
{
	for (int i = 0; i < m_input_size; ++i)
	{
		m_vWeight[i] += _a * Sigmoid_Derivative(m_LastV) * _e * _Train_Data[i];
	}

	m_dBias += _a * Sigmoid_Derivative(m_LastV) * _e;
	m_LastD = Sigmoid_Derivative(m_LastV) * _e;
}

void ML_Neuron::Reset()
{
	m_dBias = -1;

	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	for (int i = 0; i < m_input_size; ++i)
	{
		m_vWeight[i] = dist(random);
	}
}

/*
* ML_Netwrok Class 시작 부분
*/

ML_Network::ML_Network(const vector<int>& _layers)
{
	for (int i = 1; i < _layers.size(); ++i)
	{
		vector<ML_Neuron> vLayers;

		for (int j = 0; j < _layers[i]; ++j)
		{
			vLayers.push_back(ML_Neuron(_layers[i - 1]));
		}

		m_vLayers.push_back(vLayers);
	}
}

vector<double> ML_Network::Calculate(const vector<double>& _vx)
{
	vector<double> vResult;
	vector<double> v_x_next_layer = _vx;

	for (int i = 0; i < m_vLayers.size(); ++i)
	{
		vResult.clear();

		for (int j = 0; j < m_vLayers[i].size(); ++j)
		{
			vResult.push_back(m_vLayers[i][j].Calculate(v_x_next_layer));
		}

		v_x_next_layer = vResult;
	}

	return vResult;
}

void ML_Network::Train_Network(int _TrainNum, double _a, const vector<pair<vector<double>, vector<double>>>& _Train_Data)
{
	for (int i = 0; i < _TrainNum; ++i)
	{
		for (int j = 0; j < _Train_Data.size(); ++j)
		{
			//cout << i << " " << j << endl;
			vector<double> o = Calculate(_Train_Data[j].first);
			vector<double> e;

			for (int k = 0; k < o.size(); ++k)
			{
				e.push_back(_Train_Data[j].second[k] - o[k]);
			}
			
			vector<double> vLastD;
			for (int k = 0; k < m_vLayers[m_vLayers.size() - 1].size(); ++k)
			{
				m_vLayers[m_vLayers.size() - 1][k].Train_Neuron(_a, e[k], m_vLayers[m_vLayers.size() - 1][k].GetLastX());
				vLastD.push_back(m_vLayers[m_vLayers.size() - 1][k].GetLastD());
			}

			if (m_vLayers.size() == 1)
				continue;

			for (int k = m_vLayers.size() - 2; k >= 0; --k)
			{
				vector<double> new_vLastD;

				for (int l = 0; l < m_vLayers[k].size(); ++l)
				{
					vector<double> vLink_w;

					for (int m = 0; m < m_vLayers[k + 1].size(); ++m)
					{
						vLink_w.push_back(m_vLayers[k + 1][m].GetWeights()[l]);
					}

					double d_e_hidden = 0.0f;

					for (int m = 0; m < vLink_w.size(); ++m)
					{
						d_e_hidden += vLink_w[m] * vLastD[m];
					}

					m_vLayers[k][l].Train_Neuron(_a, d_e_hidden, m_vLayers[k][l].GetLastX());
					new_vLastD.push_back(m_vLayers[k][l].GetLastD());
				}

				if (k == 0)
					break;

				vLastD = new_vLastD;
			}
		}
	}
}
