#include <iostream>
#include <cuda_runtime.h>

#include "Perceptron.h"

using namespace std;

#define DATA_NUM 4
#define WEIGHT_NUM 3

__global__ void printHelloCUDA()
{
    printf("Hello CUDA!\n");
}

int main()
{
    float fe = 0.25; //학습률
    //float fx[DATA_NUM][WEIGHT_NUM] = { {1, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 1} }; //입력 신호
    float fx[DATA_NUM][WEIGHT_NUM] = { {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1} }; //입력 신호

    float ft[DATA_NUM] = {0, 1, 1, 1}; //논리합
    //float ft[DATA_NUM] = {0, 0, 0곱, 1}; //논리
    float fw[WEIGHT_NUM] = {0, 0, 0}; //가중치 초기화

    int iEpoch = 10;
    for (int i = 0; i < iEpoch; i++)
    {
        cout << "횟수: " << i << endl;

        for (int j = 0; j < DATA_NUM; j++)
            Perceptron::Train(fw, fx[j], ft[j], fe, WEIGHT_NUM);

        for (int j = 0; j < WEIGHT_NUM; j++)
            cout << "가중치: " << j << " | " << fw[j] << endl;

        cout << endl;
    } 

    for (int i = 0; i < DATA_NUM; i++)
        cout << Perceptron::Forward(fx[i], fw, WEIGHT_NUM) << " ";

    cout << endl;

    return 0;
}
