#include <iostream>
#include <cuda_runtime.h>

#include "Single_Layer_Perceptron.h"

using namespace std;

#define DATA_NUM 4
#define WEIGHT_NUM 3

int main()
{
	Neuron Plus_Neuron(2);

	//Plus_Neuron.Train(1000, 0.1f,
	//	{
	//		//트레이닝셋(지도학습)
	//		{vector<double>({0, 0}).data(), 0},
	//		{vector<double>({1, 0}).data(), 1},
	//		{vector<double>({2, 1}).data(), 1},
	//		{vector<double>({3, 1}).data(), 2},
	//		{vector<double>({2, 2}).data(), 0},
	//		{vector<double>({3, 2}).data(), 1},
	//		{vector<double>({8, 5}).data(), 3}
	//	});

	Plus_Neuron.Train(1, 0.1f,
		{
			//트레이닝셋(지도학습)
			{{0, 0}, 0},
			{{1, 0}, 1},
			{{2, 1}, 1},
			{{3, 1}, 2},
			{{2, 2}, 0},
			{{3, 2}, 1},
			{{8, 5}, 3}
		});

	std::cout.setf(ios::fixed);
	std::cout.precision(5);
	std::cout << "0 - 0 = " << Plus_Neuron.Calculate({ 0, 0 }) << endl;
	std::cout << "1 - 0 = " << Plus_Neuron.Calculate({ 1, 0 }) << endl;
	std::cout << "2 - 1 = " << Plus_Neuron.Calculate({ 2, 1 }) << endl;
	std::cout << "3 - 1 = " << Plus_Neuron.Calculate({ 3, 1 }) << endl;
	std::cout << "4 - 2 = " << Plus_Neuron.Calculate({ 4, 2 }) << endl;
	std::cout << "50 - 5 = " << Plus_Neuron.Calculate({ 50, 5 }) << endl;

	return 0;
}
