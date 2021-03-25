#include "LSTM_Layer.h"

__global__ void CUDA_LSTM_CalGate(double* _XWeight, double* _HWeight, double _H, double _Bias, double _Input, double* Result)
{
	int x = threadIdx.x;

	Result[x] = _XWeight[x] * _Input + _HWeight[x] * _H + _Bias;
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
	}
}

//Many2One
vector<double> LSTM_Layer::Calculate_M2O(double _C, double _H, const vector<double>& _InputData)
{
	static vector<double> dOutput;
	static int Count = 0;

	if (_InputData.size() <= Count)
	{
		dOutput.push_back(_H);

		return dOutput;
	}

	double g, i, o, f, c, h;

	/*
	* Tanh는 ReLU로 바꿔서 사용할 수 있음
	*/

	//입력 게이트
	i = Sigmoid(m_dXWeight[0] * _InputData[Count] + m_dHWeight[0] * _H + m_dBias[0]);
	g = Tanh(m_dXWeight[1] * _InputData[Count] + m_dHWeight[1] * _H + m_dBias[1]);

	//삭제 게이트
	f = Sigmoid(m_dXWeight[2] * _InputData[Count] + m_dHWeight[2] * _H + m_dBias[2]);

	//셀
	c = f * _C + i * g;

	//출력
	o = Sigmoid(m_dXWeight[3] * _InputData[Count] + m_dHWeight[3] * _H + m_dBias[3]);
	h = o * Tanh(c);

	Count++;

	Calculate_M2O(c, h, _InputData);

	return dOutput;
}

LSTM_Network::LSTM_Network()
{
}

vector<double> LSTM_Network::Calculate(const vector<double>& _InputData)
{
	return m_Layer.Calculate_M2O(0, 0, _InputData);
}
