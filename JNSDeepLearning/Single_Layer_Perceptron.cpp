#include "Single_Layer_Perceptron.h"

__device__ double ReLU_Device(int _x)
{
	return _x > 0 ? _x : 0;
}

__global__ void Trainning(size_t _input_size, double _a, double* _dBias,
	double* _vWeight, double* _TrainDataFirst, double* _TrainDataSecond)
{
	int j = blockIdx.y;

	int index = (2 * j);

	double t = (_TrainDataSecond[j]);

	double wx = 0.0;
	for (size_t k = 0; k < _input_size; ++k)
	{
		wx += _vWeight[k] * (_TrainDataFirst[index + k]);
	}

	double o = ReLU_Device(wx + _dBias[0]);

	for (size_t k = 0; k < _input_size; ++k)
	{
		_vWeight[k] += _a * (t - o) * (_TrainDataFirst[index + k]);
	}

	_dBias[0] += _a * (t - o);

	printf("%d %f %f %f %f\n", index, _dBias[0], _TrainDataFirst[index], _TrainDataFirst[index + 1], t);
}

__global__ void printHelloCUDA()
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	int k = blockIdx.x;
	int l = blockIdx.y;

	printf("%d %d %d %d\n", k, l, i, j);
}

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

	double* vWeight;
	cudaMalloc((void**)&vWeight, sizeof(double) * m_input_size);
	cudaMemcpy(vWeight, m_vWeight, sizeof(double) * m_input_size, cudaMemcpyHostToDevice);

	double* dTrainFirst;
	cudaMalloc((void**)&dTrainFirst, sizeof(double) * m_input_size * _train_data.size());

	double* dTrainSecond;
	cudaMalloc((void**)&dTrainSecond, sizeof(double) * _train_data.size());

	double* dstTrainFirst = dTrainFirst;
	double* dstTrainSecond = dTrainSecond;
	for (int i = 0; i < _train_data.size(); i++)
	{
		for (vector<double>::iterator iter = _train_data[i].first.begin(); iter != _train_data[i].first.end(); ++iter)
		{
			double* DTF = &((*iter));
			cudaMemcpy(dstTrainFirst, DTF, sizeof(double), cudaMemcpyHostToDevice);
			dstTrainFirst += 1;
		}

		cudaMemcpy(dstTrainSecond, &_train_data[i].second, sizeof(double), cudaMemcpyHostToDevice);
		dstTrainSecond += 1;
	}

	double* dBias;
	cudaMalloc((void**)&dBias, sizeof(double));
	cudaMemcpy(dBias, &m_dBias, sizeof(double), cudaMemcpyHostToDevice);

	dim3 threads(150, _train_data.size());
	//dim3 threads(1, _train_data.size());
	//Trainning << <_train_num, threads >> > (m_input_size, _a, dBias, vWeight, dTrainFirst, dTrainSecond);
	Trainning << <threads, 1 >> > (m_input_size, _a, dBias, vWeight, dTrainFirst, dTrainSecond);

	cudaMemcpy(&m_dBias, dBias, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_vWeight, vWeight, sizeof(double) * 2, cudaMemcpyDeviceToHost);

	//cout << " Bidas: " << m_dBias << endl;

	cudaFree(vWeight);
	cudaFree(dTrainFirst);
	cudaFree(dTrainSecond);
	cudaFree(dBias);
}

void Neuron::Test()
{

}
