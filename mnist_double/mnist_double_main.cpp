#include "mnist.h"
#include "mnist_debug.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE 0.001
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
	Network() : hiddenLayer(128, TOTAL_PIXEL), outputLayer(10, 128)
	{
		int n1 = TOTAL_PIXEL, n2 = hiddenLayer.neuronNum, n3 = outputLayer.neuronNum;
		theta2 = new double[n2], theta3 = new double[n3], delta1 = new double* [n1], delta2 = new double* [n2], answers = new double[n3];
		for (int i = 0; i < n1; i++)
		{
			delta1[i] = new double[n2];
			for (int j = 0; j < n2; j++) delta1[i][j] = 0;
		}
		for (int i = 0; i < n2; i++)
		{
			delta2[i] = new double[n3];
			for (int j = 0; j < n3; j++) delta2[i][j] = 0;
		}
	}
	~Network()
	{
		delete[] theta2, theta3, delta1, delta2, answers;
	}
	void Forward()
	{
		for (int i = 0; i < hiddenLayer.neuronNum; i++) hiddenLayer.activation[i] = 0.0;
		for (int i = 0; i < TOTAL_PIXEL; i++) for (int j = 0; j < hiddenLayer.neuronNum; j++) hiddenLayer.activation[j] += hiddenLayer.weight[i][j] * ((double)Data::image[i] / MAX_COLOR_VALUE);
		for (int i = 0; i < hiddenLayer.neuronNum; i++) hiddenLayer.activation[i] = Sigmoid(hiddenLayer.activation[i]);
		for (int i = 0; i < outputLayer.neuronNum; i++) outputLayer.activation[i] = 0.0;
		for (int i = 0; i < hiddenLayer.neuronNum; i++) for (int j = 0; j < outputLayer.neuronNum; j++) outputLayer.activation[j] += outputLayer.weight[i][j] * hiddenLayer.activation[i];
		for (int i = 0; i < outputLayer.neuronNum; i++) outputLayer.activation[i] = Sigmoid(outputLayer.activation[i]);
	}
	void Backward()
	{
		int n1 = TOTAL_PIXEL, n2 = hiddenLayer.neuronNum, n3 = outputLayer.neuronNum;
		double sum = 0.0;
		for (int i = 0; i < n3; i++) answers[i] = 0; answers[Data::label] = 1;
		for (int i = 0; i < n3; i++) theta3[i] = outputLayer.activation[i] * (1 - outputLayer.activation[i]) * (answers[i] - outputLayer.activation[i]);
		for (int i = 0; i < n2; i++)
		{
			sum = 0.0;
			for (int j = 0; j < n3; j++) sum += outputLayer.weight[i][j] * theta3[j];
			theta2[i] = hiddenLayer.activation[i] * (1 - hiddenLayer.activation[i]) * sum;
		}
		for (int i = 0; i < n2; i++)
		{
			for (int j = 0; j < n3; j++)
			{
				delta2[i][j] = (L_RATE * theta3[j] * hiddenLayer.activation[i]) + (MOMENTUM * delta2[i][j]);
				outputLayer.weight[i][j] += delta2[i][j];
			}
		}
		for (int i = 0; i < n1; i++)
		{
			for (int j = 0; j < n2; j++)
			{
				delta1[i][j] = (L_RATE * theta2[j] * (double)Data::image[i]) + (MOMENTUM * delta1[i][j]);
				hiddenLayer.weight[i][j] += delta1[i][j];
			}
		}
	}
	void FindAnswer()
	{
		int guessAnswer = 0;
		for (int i = 0; i < outputLayer.neuronNum; i++) if (outputLayer.activation[i] > outputLayer.activation[guessAnswer]) guessAnswer = i;
		fprintf(f, "======== answer = %d   guess = %d   ", Data::label, guessAnswer);
		printf("======== answer = %d   guess = %d\n", Data::label, guessAnswer);
		for (int j = 0; j < 10; j++) fprintf(f, "%f ", outputLayer.activation[j]);
		fprintf(f, "\n");
	}
	void FindAnswer(double& cnt)
	{
		int guessAnswer = 0;
		for (int i = 0; i < outputLayer.neuronNum; i++) if (outputLayer.activation[i] > outputLayer.activation[guessAnswer]) guessAnswer = i;
		if (guessAnswer == Data::label) cnt++;
	}
protected:
	Layer hiddenLayer, outputLayer;
	double _inline Sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}
	double * theta2, * theta3, ** delta1, ** delta2, * answers;
} network;
int main()
{
	_FOPEN;
	srand((unsigned int)time(NULL)); rand();
	Data::ResetData(true);
	for (int i = 0; i < 60000; i++)
	{
		printf("train %d ", i);
		Data::ReadNextTrain();
		network.Forward();
		network.Backward();
		network.FindAnswer();
	}
	Data::ResetData(false);
	double cnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		Data::ReadNextTest();
		network.Forward();
		network.FindAnswer(cnt);
	}
	printf("¥¿½T²v =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	_FCLOSE;
	return 0;
}
