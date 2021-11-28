#pragma once
#include "../LSTM/LSTM_Network.h"
#include <Activation_Function.h>

using namespace Action_Function;

template <typename T>
class BiLSTM_Network
{
public:
	BiLSTM_Network()
	{

	}
	~BiLSTM_Network() {};

	void Calculate(const vector<T> _InputData, vector<T>& _Answer)
	{
		vector<T> Forward_Answer;
		vector<T> Backward_Answer;

		Forward_LSTM.Calculate(_InputData, Forward_Answer);
		Backward_LSTM.Calculate(Vector_Reverse_Function(_InputData), Backward_Answer);
		
		//Attention Mechanism ÀÛ¾÷
	}

	void Train(const vector<T> _InputData, const vector<T> _Answer)
	{
		vector<T> Forward_Answer;
		vector<T> Backward_Answer;

		Forward_LSTM.Calculate(_InputData, Forward_Answer);
		Backward_LSTM.Calculate(Vector_Reverse_Function(_InputData), Backward_Answer);

		
	}

private:
	LSTM_Neuron<T> Forward_LSTM;
	LSTM_Neuron<T> Backward_LSTM;
};