#include "LSTM_Layer.h"

__global__ void CUDA_LSTM_CalGate(double* _XWeight, double* _HWeight, double* _Bias, double _H, double _Input, double* Result)
{
	int x = threadIdx.x;

	Result[x] = _XWeight[x] * _Input + _HWeight[x] * _H + _Bias[x];
}

LSTM_Layer::LSTM_Layer()
{
	random_device rd;
	mt19937 random(rd());
	uniform_real_distribution<double> dist(-1, 1);

	for (int i = 0; i < 4; ++i)
	{
		m_dXWeight[i] = dist(random);
		m_dHWeight[i] = dist(random);
		m_dBias[i] = -1;
	}
}

//Many2One
double LSTM_Layer::Calculate_M2O(double _C, double _H, const vector<double>& _InputData)
{
	static double dOutput;
	static int Count = 0;

	//double g, i, o, f, c, h;
	static double* Gate = (double*)malloc(sizeof(double) * 4);

	if (_InputData.size() <= Count)
	{
		Count = 0;

		double v = m_VWeight * _H + m_VBias;
		dOutput = v;

		return dOutput;
	}

	/*
	* Tanh는 ReLU로 바꿔서 사용할 수 있음
	*/

	double* pGate, * pXWeight, * pHWeight, * pBias = 0;
	cudaMalloc((void**)&pGate, sizeof(double) * 4);
	cudaMalloc((void**)&pXWeight, sizeof(double) * 4);
	cudaMalloc((void**)&pHWeight, sizeof(double) * 4);
	cudaMalloc((void**)&pBias, sizeof(double) * 4);

	cudaMemcpy(pXWeight, m_dXWeight, sizeof(double) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(pHWeight, m_dHWeight, sizeof(double) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(pBias, m_dBias, sizeof(double) * 4, cudaMemcpyHostToDevice);

	CUDA_LSTM_CalGate << <1, 4 >> > (pXWeight, pHWeight, pBias, _H, _InputData[Count], pGate);

	cudaMemcpy(Gate, pGate, sizeof(double) * 4, cudaMemcpyDeviceToHost);

	c = Gate[2] * _C + Gate[0] * Gate[1];
	h = Gate[3] * Tanh(c);

	Mem_CH.push_back(pair<double, double>(c, h));

	Count++;

	Calculate_M2O(c, h, _InputData);

	return dOutput;
}

void LSTM_Layer::BackWardPass_M2O(double _C, double _H, double _dV, const vector<double>& _InputData)
{
	//시작 C,H는 어떻게 할지 
}

void LSTM_Layer::Train_M2O(double _e, double _a, const vector<double>& _TrainData)
{
	BackWardPass_M2O(c, h, _e, _TrainData);

	Mem_CH.clear();
}

LSTM_Network::LSTM_Network()
{
}

double LSTM_Network::Calculate_M2O(const vector<double>& _InputData)
{
	return m_Layer.Calculate_M2O(0, 0, _InputData);
}

void LSTM_Network::Train_M2O(const vector<pair<vector<double>, double>>& _TrainData)
{
	for (int i = 0; i < _TrainData.size(); ++i)
	{
		double e = _TrainData[i].second - m_Layer.Calculate_M2O(0, 0, _TrainData[i].first);

		m_Layer.Train_M2O(e, 0.1, _TrainData[i].first);
	}
}