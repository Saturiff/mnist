#include "rnn.h"
#ifdef RNN_MAIN

#include "mnist_debug.h"
#include "ReinaLibrary.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE 0.0015 // 90.82
#define MOMENTUM 0.9
class Layer
{
protected:
	Layer(int neuronNum, int inputPreNeuron)
	{
		this->neuronNum = neuronNum;
		this->inputPreNeuron = inputPreNeuron;
		weight = new double* [inputPreNeuron];
		
		activation = new double[neuronNum];
		for (int i = 0; i < neuronNum; i++) activation[i] = 0;
		for (int i = 0; i < inputPreNeuron; i++)
		{
			weight[i] = new double[neuronNum];
			for (int j = 0; j < neuronNum; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
		}
	}
	~Layer()
	{
		delete[] weight, activation;
	}
	double** weight, * activation;
	int neuronNum, inputPreNeuron;
	friend class Network;
};
class Network
{
public:
	Network() : hiddenLayer(128, 7), outputLayer(10, 128)
	{
		n1 = 7, n2 = hiddenLayer.neuronNum, n3 = outputLayer.neuronNum;
		theta2 = new double[n2], theta3 = new double[n3], delta1 = new double* [n1], delta2 = new double* [n2], answers = new double[n3], mem = new double[n2], memWeight = new double[n2];
		for (int i = 0; i < n1; i++)
		{
			delta1[i] = new double[n2];
			for (int j = 0; j < n2; j++) delta1[i][j] = 0;
		}
		for (int i = 0; i < n2; i++)
		{
			delta2[i] = new double[n3];
			for (int j = 0; j < n3; j++) delta2[i][j] = 0;
			memWeight[i] = rand() * 1.0 / RAND_MAX - 0.5;
		}
	}
	~Network()
	{
		delete[] theta2, theta3, delta1, delta2, answers, mem;
	}
	void Forward()
	{
		for (int i = 0; i < n2; i++) hiddenLayer.activation[i] = 0.0;
		for (int i = 0; i < n1; i++) for (int j = 0; j < n2; j++) hiddenLayer.activation[j] += hiddenLayer.weight[i][j] * (double)Data::input[i] + memWeight[j] * mem[j];
		for (int i = 0; i < n2; i++) hiddenLayer.activation[i] = mem[i] = Sigmoid(hiddenLayer.activation[i]);
		for (int i = 0; i < n3; i++) outputLayer.activation[i] = 0.0;
		for (int i = 0; i < n2; i++) for (int j = 0; j < n3; j++) outputLayer.activation[j] += outputLayer.weight[i][j] * hiddenLayer.activation[i];
		for (int i = 0; i < n3; i++) outputLayer.activation[i] = Sigmoid(outputLayer.activation[i]);
	}
	void Backward()
	{
		double sum = 0.0;
		for (int i = 0; i < n3; i++) answers[i] = 0; answers[Data::ans] = 1;
		for (int i = 0; i < n3; i++) theta3[i] = SigmoidDerivative(outputLayer.activation[i]) * (answers[i] - outputLayer.activation[i]);
		for (int i = 0; i < n2; i++)
		{
			sum = 0.0;
			for (int j = 0; j < n3; j++) sum += outputLayer.weight[i][j] * theta3[j];
			theta2[i] = SigmoidDerivative(hiddenLayer.activation[i]) * sum;
		}
		for (int i = 0; i < n2; i++)
		for (int j = 0; j < n3; j++)
		{
			delta2[i][j] = (L_RATE * theta3[j] * hiddenLayer.activation[i]) + (MOMENTUM * delta2[i][j]);
			outputLayer.weight[i][j] += delta2[i][j];
		}
		for (int i = 0; i < n1; i++)
		for (int j = 0; j < n2; j++)
		{
			delta1[i][j] = (L_RATE * theta2[j] * (double)Data::input[i]) + (MOMENTUM * delta1[i][j]);
			hiddenLayer.weight[i][j] += delta1[i][j];
		}
		for (int i = 0; i < n2; i++)
		{
			memWeight[i] += L_RATE * theta2[i] * mem[i];
		}
	}
	void ResetMem()
	{
		for (int i = 0; i < n2; i++) mem[i] = 0;
	}
	void FindAnswer(double& cnt)
	{
		int guessAnswer = 0;
		for (int i = 0; i < n3; i++) if (outputLayer.activation[i] > outputLayer.activation[guessAnswer]) guessAnswer = i;
		if (guessAnswer == Data::ans) cnt++;
	}
protected:
	double _inline Sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}
	double _inline SigmoidDerivative(double x)
	{
		return x * (1 - x);
	}
	Layer hiddenLayer, outputLayer;
	int n1, n2, n3;
	double* theta2, * theta3, ** delta1, ** delta2, * answers, * mem, * memWeight;
} network;
int main()
{
	_FOPEN;
	srand((unsigned int)time(NULL)); rand();
	Data::ResetData();
	for (int i = 0; i < TRAIN_ITEMS; i++) // 資料數
	{
		for (int j = 0; j < 7; j++)
		{
			Data::ReadNextTrain();
			network.Forward();
			network.Backward();
		}
		network.ResetMem();
	}
	Data::ResetData();
	double cnt = 0, localCnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		localCnt = 0;
		for (int j = 0; j < 7; j++)
		{
			Data::ReadNextTest();
			network.Forward();
			network.FindAnswer(localCnt);
		}
		if (localCnt == 7) cnt += 1;
		network.ResetMem();
	}
	printf("正確率 =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	_FCLOSE;
	return 0;
}

#endif // RNN_MAIN
