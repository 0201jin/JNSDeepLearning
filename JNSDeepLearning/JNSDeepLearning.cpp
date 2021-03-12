#include <iostream>
#include <cuda_runtime.h>

#include "Single_Layer_Perceptron.h"

using namespace std;

#define DATA_NUM 4
#define WEIGHT_NUM 3

__global__ void printHelloCUDA()
{
	printf("Hello CUDA!\n");
}

int main()
{
	/*Neuron And_Neuron(2);

	for (int i = 0; i < 10000; ++i)
	{
		And_Neuron.Train(0.1f,
			{
				//트레이닝셋(지도학습)
				{{0, 0}, 0},
				{{1, 0}, 0},
				{{0, 1}, 0},
				{{1, 1}, 1}
			});
	}

	cout.setf(ios::fixed);
	cout.precision(5);
	cout << "0 AND 0 = " << And_Neuron.Calculate({ 0, 0 }) << endl;
	cout << "1 AND 0 = " << And_Neuron.Calculate({ 1, 0 }) << endl;
	cout << "0 AND 1 = " << And_Neuron.Calculate({ 0, 1 }) << endl;
	cout << "1 AND 1 = " << And_Neuron.Calculate({ 1, 1 }) << endl;*/

	Neuron Plus_Neuron(2);

	for (int i = 0; i < 1000; ++i)
	{
		Plus_Neuron.Train(0.1f,
			{
				//트레이닝셋(지도학습)
				{{0, 0}, 0},
				{{1, 0}, 1},
				{{2, 1}, 3},
				{{3, 1}, 4}
			});
	}

	cout.setf(ios::fixed);
	cout.precision(5);
	cout << "0 + 0 = " << Plus_Neuron.Calculate({ 0, 0 }) << endl;
	cout << "1 + 0 = " << Plus_Neuron.Calculate({ 1, 0 }) << endl;
	cout << "2 + 1 = " << Plus_Neuron.Calculate({ 2, 1 }) << endl;
	cout << "3 + 1 = " << Plus_Neuron.Calculate({ 3, 1 }) << endl;
	cout << "2 + 2 = " << Plus_Neuron.Calculate({ 4, 2 }) << endl;
	cout << "50 + 5 = " << Plus_Neuron.Calculate({ 50, 5 }) << endl;

	return 0;
}
