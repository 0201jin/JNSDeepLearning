﻿#include <iostream>
#include <fstream>

#include "Single_Layer_Perceptron.h"
#include "Multi_Layer_Perceptron/Multi_Layer_Perceptron.h"
#include "RNN/RNN.h"
#include "LSTM/LSTM/LSTM_Network.h"
#include "LSTM/BiLSTM/BiLSTM.h"
#include <time.h>
#include <chrono>

using namespace std;

#define DATA_NUM 4
#define WEIGHT_NUM 3
//코드를 수정합니다.

void Single_Neuron_Run()
{
	Single_Neuron Plus_Neuron(2);

	Plus_Neuron.Train(100000, 0.1f,
		{
			//트레이닝셋(지도학습)
			{{0, 0}, 0},
			{{1, 0}, 1},
			{{2, 1}, 3},
			{{3, 1}, 4},
			{{2, 2}, 4},
			{{3, 2}, 5},
			{{8, 5}, 13}
		});

	cout.setf(ios::fixed);
	cout.precision(10);
	cout << "0 + 0 = " << Plus_Neuron.Calculate({ 0, 0 }) << endl;
	cout << "1 + 0 = " << Plus_Neuron.Calculate({ 1, 0 }) << endl;
	cout << "2 + 1 = " << Plus_Neuron.Calculate({ 2, 1 }) << endl;
	cout << "3 + 1 = " << Plus_Neuron.Calculate({ 3, 1 }) << endl;
	cout << "4 + 2 = " << Plus_Neuron.Calculate({ 4, 2 }) << endl;
	cout << "50 + 5 = " << Plus_Neuron.Calculate({ 50, 5 }) << endl;
	cout << "150 + 5 = " << Plus_Neuron.Calculate({ 150, 5 }) << endl;
}

void Multi_Nueron_Run()
{
	ML_Network net({ 2, 4, 4, 4, 1 });

	net.Train_Network(10000, 0.1,
		{
			{ { 1, 0 }, { 1 } },
			{ { 1, 1 }, { 1 } },
			{ { 0, 1 }, { 0 } },
			{ { 1, 0 }, { 0 } },
		});

	printf("0 == 1 = %f\n", net.Calculate({ 0, 1 })[0]);
	printf("%d\n", 'A');
	/*printf("0 xor 0 = %f %f\n", net.Calculate({ 0, 0 })[0], net.Calculate({ 0, 0 })[1]);
	printf("1 xor 0 = %f %f\n", net.Calculate({ 1, 0 })[0], net.Calculate({ 1, 0 })[1]);
	printf("0 xor 1 = %f %f\n", net.Calculate({ 0, 1 })[0], net.Calculate({ 0, 1 })[1]);
	printf("1 xor 1 = %f %f\n", net.Calculate({ 1, 1 })[0], net.Calculate({ 1, 1 })[1]);
	printf("2 xor 1 = %f %f\n", net.Calculate({ 2, 1 })[0], net.Calculate({ 2, 1 })[1]);*/
}

void RNN_Run()
{
	RNN_Network net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train_M2O({ {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8} }, { 0.4, 0.7, 0.9 });
	}

	printf("%f\n", net.Calculate_M2O({ 0.3, 0.4, 0.5 }));
}

void RNN_O2M_Run()
{
	RNN_Network net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train_O2M({ 0.1, 0.2, 0.3 , 0.4, 0.5 }, { {0.2, 0.3, 0.4}, {0.3, 0.4, 0.5}, {0.4, 0.5, 0.6}, {0.5, 0.6, 0.7}, {0.6, 0.7, 0.8} });
	}

	net.Calculate_O2M(0.3f);
	net.printY();
}

void RNN_M2M_Run()
{
	RNN_Network net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train_M2M({ {0.1, 0.2, 0.3}, {0.2, 0.3, 0.4}, {0.3, 0.4, 0.5} },
			{ {0.4, 0.5, 0.6}, {0.5, 0.6, 0.7}, {0.6, 0.7, 0.8} });
	}

	net.Calculate_M2M({ 0.4, 0.5, 0.6 });
	net.printY();
}

void LSTM_M2O_Run()
{
	LSTM_Network<double> net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train(
			{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
			{ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 },
			5);
	}

	double Answer;
	net.Calculate({ 0.1, 0.2, 0.3, 0.4, 0.5 }, Answer);
	cout << Answer << endl;
}

void LSTM_M2M_Run()
{
	LSTM_Network<double> net;

	for (int i = 0; i < 20000; ++i)
	{
		net.TrainM2M(
			{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
			{ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 },
			5);
	}

	vector<double> Answer;
	net.Calculate({ 0.4, 0.5, 0.6, 0.7, 0.8 }, Answer);
	
	for (int i = 0; i < Answer.size(); ++i)
		cout << Answer[i] << endl;
}

void BILSTM_M2M_Run()
{
	BiLSTM_Network<double> net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train(
			{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 },
			{ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4}, 5);
	}

	vector<double> Answer;
	net.Calculate({ 1.0, 1.1, 1.2, 1.3 }, Answer);

	for (int i = 0; i < Answer.size(); ++i)
		cout << Answer[i] << endl;
}

void BILSTM_M2M_UseDataSet_Run(const string _DataSetPath)
{
	ifstream DataSet(_DataSetPath);

	if(DataSet.fail())
	{
		cout << "DataSet Can't Load!"<< endl;
		return;
	}

	BiLSTM_Network<double> net;

	for(int i = 0; i < 1; ++i)
	{
		net.Train(DataSet, 1);
	}

	DataSet.close();
}

int main()
{
	auto start = chrono::system_clock::now();

	//RNN_M2M_Run();
	//LSTM_M2O_Run();
	//LSTM_M2M_Run();
	//BILSTM_M2M_Run();
	BILSTM_M2M_UseDataSet_Run("all.csv");

	auto end = chrono::system_clock::now();

	chrono::microseconds delta = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "실행 속도 : " << delta.count() << endl;

	return 0;
}
