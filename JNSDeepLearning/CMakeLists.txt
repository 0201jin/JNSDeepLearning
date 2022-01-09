project(JNSDeepLearning LANGUAGES CXX CUDA)

ADD_EXECUTABLE( JNSDeepLearning 
JNSDeepLearning.cpp 
Single_Layer_Perceptron.cpp 
LSTM/BiLSTM/BiLSTM.cpp 
LSTM/LSTM/LSTM_Network.cpp 
Multi_Layer_Perceptron/Multi_Layer_Perceptron.cpp 
RNN/RNN.cpp 
CUDA/CUDA_Matrix.cu 
)
