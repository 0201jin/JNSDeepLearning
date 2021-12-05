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
		random_device rd;
		mt19937 random(rd());
		uniform_real_distribution<double> dist(-1, 1);

		d_Weight = dist(random);
		d_Bias = dist(random);
	}
	~BiLSTM_Network() {};

	void Calculate(const vector<T> _InputData, vector<T>& _Answer)
	{
		vector<T> Forward_Answer;
		vector<T> Backward_Answer;

		Forward_LSTM.Calculate(_InputData, Forward_Answer);
		Backward_LSTM.Calculate(Vector_Reverse_Function(_InputData), Backward_Answer);

		for (int i = 0; i < Forward_Answer.size(); ++i)
			_Answer.push_back(Forward_Answer[i] * Backward_Answer[i]);
		
		//Attention Mechanism ÀÛ¾÷
	}

	void Train(const vector<T> _InputData, const vector<T> _Answer, int _Sequence)
	{
		for (int i = 0; i < _InputData.size() - (_Sequence - 1); ++i)
		{
			int Index = i + _Sequence - 1;
			vector<T> TrainData(_Sequence);
			vector<T> AnswerData(_Sequence);

			copy(_InputData.begin() + i, _InputData.begin() + Index + 1, TrainData.begin());
			copy(_Answer.begin() + i, _Answer.begin() + Index + 1, AnswerData.begin());

			Forward_LSTM.Train(TrainData, AnswerData);
			Backward_LSTM.Train(Vector_Reverse_Function(TrainData), AnswerData);
		}
	}

private:
	LSTM_Neuron<T> Forward_LSTM;
	LSTM_Neuron<T> Backward_LSTM;

	double d_Weight;
	double d_Bias;
};