#include <iostream>
#include <cuda_runtime.h>

#include "Single_Layer_Perceptron.h"
#include "Multi_Layer_Perceptron/Multi_Layer_Perceptron.h"
#include "LSTM/LSTM_Layer/LSTM_Layer.h"
#include "RNN/RNN.h"

using namespace std;

#define DATA_NUM 4
#define WEIGHT_NUM 3

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
	cout << "50 + 5 = " << Plus_Neuron.Calculate({ 150, 5 }) << endl;
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

void LSTM_Run()
{
	LSTM_Network net;

	for (int i = 0; i < 1000; ++i)
		net.Train_M2O(
			{
				{{1, 1}, 1 }
			}
	);

	printf("%f", net.Calculate_M2O({ 1, 1 }));
}

void RNN_Run()
{
	RNN_Network net;

	for (int i = 0; i < 10000; ++i)
	{
		net.Train_M2O({ { 1, 2, 3, 4 } }, { 5 });
	}

	printf("%f", net.Calculate_M2O({ 5, 6, 7, 8 }));
}

int main()
{
	RNN_Run();

	return 0;
}
