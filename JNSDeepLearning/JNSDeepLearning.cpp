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
	Neuron And_Neuron(2);

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
	cout << "1 AND 1 = " << And_Neuron.Calculate({ 1, 1 }) << endl;

	return 0;
}
