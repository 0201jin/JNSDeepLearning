#pragma once
#include <fstream>
#include <queue>
#include "../LSTM/LSTM_Network.h"
#include "../../Activation_Function.h"
#include "../../Technical_Function.h"

using namespace std;
using namespace Action_Function;
using namespace Technical_Function;

template <typename T>
class BiLSTM_Network
{
public:
	BiLSTM_Network()
	{
		random_device rd;
		mt19937 random(rd());
		uniform_real_distribution<double> dist(-1, 1);

		d_FWeight = dist(random);
		d_BWeight = dist(random);
		d_Bias = dist(random);
	}
	~BiLSTM_Network() {};

	//M2M Custom DataSet Train
	void Calculate()
	{

	}

	void Train(ifstream& _DataSetFile, int _Sequence)
	{
		int DataSize = 2;
		int iDataSetColum = 0;

		//queue를 이용해서 데이터 셋 저장
		queue<vector<double>> DataSet; 

		string str_buf;
		while(getline(_DataSetFile, str_buf, '\n'))
		{
			vector<string> DataStringVector = SplitData(str_buf, ",");
			DataSet.push(VS2VD(DataStringVector));

			if(DataSet.size() < _Sequence)
				continue;

			DataSet.pop();
		}
	}

	//M2M
	void Calculate(const vector<T> _InputData, vector<T>& _Answer)
	{
		Forward_LSTM.Calculate(_InputData, Forward_Answer);
		Backward_LSTM.Calculate(Vector_Reverse_Function(_InputData), Backward_Answer);

		for (int i = 0; i < Forward_Answer.size(); ++i)
			_Answer.push_back(Forward_Answer[i] * d_FWeight + Backward_Answer[i] * d_BWeight + d_Bias);
		
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

			vector<T> CAnswer;
			Calculate(TrainData, CAnswer);

			vector<double> FH;
			vector<double> BH;

			for (int j = _Sequence - 1; j >= 0; --j)
			{
				double dy = 2 * (CAnswer[j] - AnswerData[j]);
				
				FH.push_back(dy * d_FWeight);
				BH.push_back(dy * d_BWeight);
				
				d_FWeight -= dy * Forward_Answer[j] * LEARN_RATE;
				d_BWeight -= dy * Backward_Answer[j] * LEARN_RATE;
				d_Bias -= dy * LEARN_RATE;
			}

			Forward_LSTM.Train(TrainData, AnswerData);
			Backward_LSTM.Train(Vector_Reverse_Function(TrainData), AnswerData);

			Forward_Answer.clear();
			Backward_Answer.clear();
		}
	}

	//NLP

private:
	vector<T> Forward_Answer;
	vector<T> Backward_Answer;

	LSTM_Neuron<T> Forward_LSTM;
	LSTM_Neuron<T> Backward_LSTM;

	double d_FWeight;
	double d_BWeight;
	double d_Bias;
};
