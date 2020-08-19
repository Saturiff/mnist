// not working!

#include "rnn.h"
#include "mnist_debug.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE 0.0001 // 
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
		theta2 = new double[n2], theta3 = new double[n3], delta1 = new double* [n1], delta2 = new double* [n2], deltaOutput = new double* [n1], deltaInput = new double* [n1], deltaForget = new double* [n1], answers = new double[n3], memory = new double[n2], inputGateW = new double* [n1], forgetGateW = new double* [n1], outputGateW = new double* [n1], inputGateA = new double[n2], forgetGateA = new double[n2], outputGateA = new double[n2];
		for (int i = 0; i < n1; i++)
		{
			delta1[i] = new double[n2];
			deltaForget[i] = new double[n2];
			deltaInput[i] = new double[n2];
			deltaOutput[i] = new double[n2];
			inputGateW[i] = new double[n2];
			forgetGateW[i] = new double[n2];
			outputGateW[i] = new double[n2];
			for (int j = 0; j < n2; j++)
			{
				delta1[i][j] = deltaForget[i][j] = deltaInput[i][j] = deltaOutput[i][j] = 0;
				inputGateW[i][j] = forgetGateW[i][j] = outputGateW[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
			}
		}
		for (int i = 0; i < n2; i++)
		{
			delta2[i] = new double[n3];
			for (int j = 0; j < n3; j++) delta2[i][j] = 0;
			inputGateA[i] = forgetGateA[i] = outputGateA[i] = 0;
		}
	}
	~Network()
	{
		delete[] theta2, theta3, delta1, delta2, answers, memory;
	}
	void Forward()
	{
		// init gate and hidden layer activation
		for (int i = 0; i < n2; i++) hiddenLayer.activation[i] = inputGateA[i] = forgetGateA[i] = outputGateA[i] = 0.0;

		// input -> hidden and gate
		for (int i = 0; i < n1; i++)
		for (int j = 0; j < n2; j++)
		{
			hiddenLayer.activation[j] += hiddenLayer.weight[i][j] * (double)Data::input[i];
			inputGateA[i] += inputGateW[i][j] * (double)Data::input[i];
			forgetGateA[i] += forgetGateW[i][j] * (double)Data::input[i];
			outputGateA[i] += outputGateW[i][j] * (double)Data::input[i];
		}
		for (int i = 0; i < n2; i++)
		{
			hiddenLayer.activation[i] = Tanh(hiddenLayer.activation[i]);
			inputGateA[i] = Sigmoid(inputGateA[i]);
			forgetGateA[i] = Sigmoid(forgetGateA[i]);
			outputGateA[i] = Sigmoid(outputGateA[i]);
		}
		for (int i = 0; i < n2; i++)
		{
			memory[i] = forgetGateA[i] * memory[i] + inputGateA[i] * hiddenLayer.activation[i];
			hiddenLayer.activation[i] = outputGateA[i] * Tanh(memory[i]);
		}

		// hidden -> output
		for (int i = 0; i < n2; i++) for (int j = 0; j < n3; j++) outputLayer.activation[j] += outputLayer.weight[i][j] * hiddenLayer.activation[i];
		for (int i = 0; i < n3; i++) outputLayer.activation[i] = Sigmoid(outputLayer.activation[i]);
	}
	void ResetMem()
	{
		for (int i = 0; i < n2; i++) memory[i] = 0;
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
			delta2[i][j] = L_RATE * theta3[j] * hiddenLayer.activation[i];
			outputLayer.weight[i][j] += delta2[i][j];
		}
		for (int i = 0; i < n1; i++)
		for (int j = 0; j < n2; j++)
		{
			deltaOutput[i][j] = L_RATE * theta2[j]  * Tanh(memory[j]) * SigmoidDerivative(outputGateA[j])           * (double)Data::input[i];
			deltaForget[i][j] = L_RATE * (theta2[j] * outputGateA[j]  * TanhDerivative(memory[j])) * outputGateA[j] * SigmoidDerivative(forgetGateA[j]) * (double)Data::input[i];
			deltaInput[i][j]  = L_RATE * (theta2[j] * outputGateA[j]  * TanhDerivative(memory[j])) * memory[j]      * SigmoidDerivative(inputGateA[j])  * (double)Data::input[i];
			delta1[i][j]      = L_RATE * (theta2[j] * outputGateA[j]  * TanhDerivative(memory[j])) * inputGateA[j]  * TanhDerivative(memory[j])         * (double)Data::input[i];

			hiddenLayer.weight[i][j] += delta1[i][j];
			inputGateW[i][j] += deltaInput[i][j];
			forgetGateW[i][j] += deltaForget[i][j];
			outputGateW[i][j] += deltaOutput[i][j];
		}
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
	double _inline Tanh(double x)
	{
		return tanh(x);
	}
	double _inline TanhDerivative(double x)
	{
		x = tanh(x);
		return 1.0 - x * x;
	}
	int n1, n2, n3;
	Layer hiddenLayer, outputLayer;
	double* theta2, * theta3, ** delta1, ** delta2, ** deltaMemory, ** deltaInput, ** deltaForget, ** deltaOutput, * answers, * memory, ** inputGateW, ** forgetGateW, ** outputGateW, * inputGateA, * forgetGateA, * outputGateA;
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
