cmake_minimum)required(VERSION 2.8)

ADD_EXECUTABLE( JNSDeepLearning 
JNSDeepLearning.cpp Single_Layer_Perceptron.cpp 
CUDA/CUDA_Matrix.c 
LSTM/BiLSTM/BiLSTM.cpp 
LSTM/LSTM_Network.cpp 
Multi_Layer_Perceptron/Multi_Layer_Perceptron.cpp 
RNN/RNN.cpp 
)
