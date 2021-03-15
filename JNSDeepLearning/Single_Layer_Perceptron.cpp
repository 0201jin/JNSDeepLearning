#include "Single_Layer_Perceptron.h"

__global__ void Trainning(size_t _input_size, double _a, double* _dBias,
	double* _vWeight, double* _TrainDataFirst, double* _TrainDataSecond)
{
	//first는 2개씩 올라가야함.
	//(second IDX = 0)  == (first IDX = 0 ~ 1)
	//(second IDX = 1)  == (first IDX = 2 ~ 3)
	int i = threadIdx.x; // _input_size
	int j = threadIdx.y; // _train_data.size()

	int index = (_input_size * j);

	double t = (_TrainDataSecond[j]);
	
	__shared__ double wx;
	wx += (_vWeight)[i] * (_TrainDataFirst[index + i]);
	
	if(i != 0)
	{
		double o = wx + (*_dBias) > 0 ? wx + (*_dBias) : 0;
		
		(_vWeight)[0] += _a * (t - o) * (_TrainDataFirst[index + 0]);
		(_vWeight)[1] += _a * (t - o) * (_TrainDataFirst[index + 1]);
		
		(*_dBias) += _a * (t - o);
		
		wx = 0;
	}

	/*double wx = 0.0;
	for (size_t k = 0; k < _input_size; ++k)
	{
		wx += (_vWeight)[k] * (_TrainDataFirst[index + k]); //k가 붙음
		printf("1 %d %d %d\n", i, j, k);
	}

	double o = wx + (*_dBias) > 0 ? wx + (*_dBias) : 0;

	for (size_t k = 0; k < _input_size; ++k)
	{
		(_vWeight)[k] += _a * (t - o) * (_TrainDataFirst[index + k]); //k가 붙음
		printf("2 %d %d %d\n", i, j, k);
	}

	(*_dBias) += _a * (t - o);*/

	//printf("%d %d\n", i, j);
}

__global__ void printHelloCUDA()
{
	int j = threadIdx.x;

	for (int i = 0; i < 100; i++)
		printf("Hello CUDA! %d\n", i + (j * 100));
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

	//dim3 threads(m_input_size, _train_data.size());
	dim3 threads(1, _train_data.size());
	Trainning << <_train_num, threads >> > (m_input_size, _a, dBias, vWeight, dTrainFirst, dTrainSecond);

	cudaFree(vWeight);
}

void Neuron::Test()
{

}
