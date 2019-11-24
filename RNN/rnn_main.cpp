#include "rnn.h"
#include "mnist_debug.h"
#include "ReinaLibrary.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
// #define ACTIONS_PRE_DATA  7
// #define STATUS_PRE_ACTION 7
// #define SEQ_ACTION        7
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
	Network() : hiddenLayer(128, 7), outputLayer(10, 128)
	{
		int n1 = 7, n2 = hiddenLayer.neuronNum, n3 = outputLayer.neuronNum;
		theta2 = new double[n2], theta3 = new double[n3], delta1 = new double* [n1], delta2 = new double* [n2], answers = new double[n3], mem = new double[n2];
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
		delete[] theta2, theta3, delta1, delta2, answers, mem;
	}
	void Forward()
	{
		for (int i = 0; i < hiddenLayer.neuronNum; i++) hiddenLayer.activation[i] = 0.0;
		//for (int i = 0; i < 7; i++) for (int j = 0; j < hiddenLayer.neuronNum; j++) hiddenLayer.activation[j] += hiddenLayer.weight[i][j] * (double)Data::input[i];
		for (int i = 0; i < 7; i++) for (int j = 0; j < hiddenLayer.neuronNum; j++) hiddenLayer.activation[j] += hiddenLayer.weight[i][j] * ((double)Data::input[i] + mem[j]);
		for (int i = 0; i < hiddenLayer.neuronNum; i++) hiddenLayer.activation[i] = mem[i] = Sigmoid(hiddenLayer.activation[i]);
		for (int i = 0; i < outputLayer.neuronNum; i++) outputLayer.activation[i] = 0.0;
		for (int i = 0; i < hiddenLayer.neuronNum; i++) for (int j = 0; j < outputLayer.neuronNum; j++) outputLayer.activation[j] += outputLayer.weight[i][j] * hiddenLayer.activation[i];
		for (int i = 0; i < outputLayer.neuronNum; i++) outputLayer.activation[i] = Sigmoid(outputLayer.activation[i]);
	}
	void Backward()
	{
		int n1 = 7, n2 = hiddenLayer.neuronNum, n3 = outputLayer.neuronNum;
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
	}
	void PrintGuess()
	{
		int guessAnswer = 0;
		for (int i = 0; i < outputLayer.neuronNum; i++) if (outputLayer.activation[i] > outputLayer.activation[guessAnswer]) guessAnswer = i;
		fprintf(f, "%d ", guessAnswer);
	}
	void FindAnswer(double& cnt)
	{
		int guessAnswer = 0;
		for (int i = 0; i < outputLayer.neuronNum; i++) if (outputLayer.activation[i] > outputLayer.activation[guessAnswer]) guessAnswer = i;
		if (guessAnswer == Data::ans) cnt++;
	}
	void PrintInput()
	{
		fprintf(f, "input seq = ");
		int status = 0;
		for (int i = 0; i < 7; i++)
		{
			if (Data::input[i]) status = i;
		}
		if (status == 0)      fprintf(f, "pay_1      ");
		else if (status == 1) fprintf(f, "pay_5      ");
		else if (status == 2) fprintf(f, "pay_10     ");
		else if (status == 3) fprintf(f, "pick_A     ");
		else if (status == 4) fprintf(f, "pick_B     ");
		else if (status == 5) fprintf(f, "pick_C     ");
		else if (status == 6) fprintf(f, "back       ");

		fprintf(f, "\n");
	}
	void PrintAns()
	{
		fprintf(f, "\nanswer seq = ");
		for (int i = 0; i < 7; i++)
		{
			if (Data::input[i] == 0 || Data::input[i] == 1 || Data::input[i] == 2)      fprintf(f, "do_nothing ");
			else if (Data::input[i] == 3 || Data::input[i] == 4 || Data::input[i] == 5) fprintf(f, "give_%c    ", 'A' + (Data::input[i] - 3));
			else if (Data::input[i] == 6)                                               fprintf(f, "back       ");
		}
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
	double* theta2, * theta3, ** delta1, ** delta2, * answers, * mem;
} network;
int main()
{
	_FOPEN;
	srand((unsigned int)time(NULL)); rand();
	Data::ResetData();
	int* in = new int[7];
	for (int i = 0; i < 60000; i++) // 資料數
	{
		//if (i > 55000)
		{
			//printf("train %d\n", i);
			//fprintf(f, "=======\ntrain %d\n", i);
			//fprintf(f, "guess = ");
			for (int j = 0; j < 7; j++)
			{
				Data::ReadNextTrain();
				Data::NextTrainAnswer();
				//network.PrintInput();
				//ConvertArrayType(Data::input, in, 7);
				//for (int i = 0; i < 7; i++) in[i] = Data::input[i];
				//Print1DArray("train input", in, 7);
				network.Forward();
				network.Backward(); 

				//fprintf(f, "\n");
				//network.PrintGuess();
			}
			
			
			
		}
	}
	printf("\n=======\n");
	Data::ResetData();
	double cnt = 0, localCnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		localCnt = 0;
		//printf("test %d\n", i);
		for (int j = 0; j < 7; j++)
		{
			Data::ReadNextTest();
			Data::NextTestAnswer();
			network.Forward();
			network.FindAnswer(localCnt);
		}
		if (localCnt == 7) cnt += 1;
	}
	printf("正確率 =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	_FCLOSE;
	return 0;
}
