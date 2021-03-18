#include "Single_Layer_Perceptron.h"

__device__ double Cuda_ReLU(double _x)
{
	return _x > 0 ? _x : 0;
}

__global__ void CUDA_Calculate(double* d_Bias, double* d_Weight, double* Train_First, double* o)
{
	int x = threadIdx.x;
	
	__shared__ double wx = 0.0;

	wx += d_Weight[x] * Train_First[x];

	__syncthreads();
	
	if(x == 0)
		o[0] = Cuda_ReLU(wx + d_Bias[0]);
}

__global__ void CUDA_CalWeight(double* dWeight, double* TrainData, double a, double t, double o)
{
	int x = threadIdx.x;
	
	dWeight[x] += a * (t - o) * TrainData[x];
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

	double* dBias;
	cudaMalloc((void**)&dBias, sizeof(double));
	cudaMemcpy(dBias, &m_dBias, sizeof(double), cudaMemcpyHostToDevice);
	
	for (size_t i = 0; i < _train_data.size(); ++i)
	{
		double t = _train_data[i].second;
		
		double o = 0;
		double* po = 0;
		cudaMalloc((void**)&po, sizeof(double));
		cudaMemcpy(po, &o, sizeof(double), cudaMemcpyHostToDevice);
		
		double* pTrainData = _train_data[i].first.data();
		
		CUDA_Calculate<<<1, m_input_size>>>(dBias, vWeight, pTrainData, po);
		
		cudaMemcpy(&o, po, sizeof(double), cudaMemcpyDeviceToHost);
		
		CUDA_CalWeight<<<1, m_input_size>>>(vWeight, pTrainData, _a, t, o);
		
		m_dBias += _a * (t - o);
	}
}

void Neuron::Test()
{

}
