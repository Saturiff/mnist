#include "mnist.h"
#include "mnist_debug.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define NEURON_QUNATITY 10

//#define L_RATE 0.09 // 91.23
//#define L_RATE 0.08 // 91.30
#define L_RATE 0.07 // 91.30
//#define L_RATE 0.06 // 91.26

double _inline Sigmoid(float x)
{
	return exp(x) / (exp(x) + 1.0);
}
double _inline ReLU(float x)
{
	return (x > 0.0) ? x : 0.0;
}
void _inline Softmax(float* in, float* out, int size)
{
	float sum = 0.0;
	for (int i = 0; i < size; i++) sum += exp(in[i]);
	for (int i = 0; i < size; i++) out[i] = exp(in[i]) / sum;
}
enum class ActiveFunction{ sigmoid, relu, softmax };
class Layer
{
public:
	Layer(int neuronNum, int inputPreNeuron, bool isInputLayer, bool isOutputLayer, ActiveFunction activeMode)
	{
		this->neuronNum = neuronNum, this->inputPreNeuron = inputPreNeuron, this->isInputLayer = isInputLayer, this->isOutputLayer = isOutputLayer, mode = activeMode;
		answers = new float[neuronNum], weight = new float* [neuronNum], bias = new float[neuronNum], activation = new float[neuronNum], z = new float[neuronNum];
		for (int i = 0; i < neuronNum; i++)
		{
			answers[i] = activation[i] = z[i] = 0;
			bias[i] = rand() * 1.0 / RAND_MAX - 0.5;
			weight[i] = new float[inputPreNeuron];
			for (int j = 0; j < inputPreNeuron; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
		}
	}
	~Layer()
	{
		delete[] answers, weight, bias, activation;
	}
	void Forward(byte* prevA)
	{
		ByteArr2FloatArr(prevA, &Layer::Forward);
	}
	void Forward(float* prevA)
	{
		for (int i = 0; i < neuronNum; i++)
		{
			z[i] = bias[i];
			for (int j = 0; j < inputPreNeuron; j++) z[i] += weight[i][j] * (prevA[j] / ((isInputLayer) ? MAX_COLOR_VALUE : 1)) * 0.001;
			if (mode == ActiveFunction::relu) activation[i] = ReLU(z[i]);
			else if (mode == ActiveFunction::sigmoid) activation[i] = Sigmoid(z[i]);
		}
		if (mode == ActiveFunction::softmax) Softmax(z, activation, neuronNum);
	}
	void SetAnswer(byte answer)
	{
		for (int i = 0; i < neuronNum; i++) answers[i] = 0; answers[answer] = 1;
	}
	void SetAnswer(float* inAnswers)
	{
		for (int i = 0; i < neuronNum; i++) answers[i] = inAnswers[i];
	}
	void Backward(byte* prevA)
	{
		ByteArr2FloatArr(prevA, &Layer::Backward);
	}
	void Backward(float* prevA)
	{
		float dy = 0.0;
		for (int i = 0; i < neuronNum; i++)
		{
			dy = outputLayer.answers[i] - outputLayer.activation[i];
			for (int j = 0; j < inputPreNeuron; j++) weight[i][j] += L_RATE * prevA[j] * dy;
			bias[i] += L_RATE * dy;
		}
	}
	void FindAnswer(byte trueAnswer)
	{
		int guessAnswer = 0;
		for (int i = 0; i < neuronNum; i++) if (activation[i] > activation[guessAnswer]) guessAnswer = i;
		fprintf(f, "======== answer = %d   guess = %d   ", Data::label, guessAnswer);
		for (int j = 0; j < 10; j++) fprintf(f, "%f ", outputLayer.activation[j]);
		fprintf(f, "\n");
	}
	void FindAnswer(byte trueAnswer, float& cnt)
	{
		int guessAnswer = 0;
		for (int i = 0; i < neuronNum; i++) if (activation[i] > activation[guessAnswer]) guessAnswer = i;
		if (guessAnswer == trueAnswer) cnt++;
		//fprintf(f, "======== answer = %d   guess = %d   ", Data::label, guessAnswer);
		//for (int j = 0; j < 10; j++) fprintf(f, "%f ", outputLayer.activation[j]);
		//fprintf(f, "\n");
	}
	float* activation,* z;
	int neuronNum, inputPreNeuron;
	bool isInputLayer, isOutputLayer;
	ActiveFunction mode;
protected:
	void ByteArr2FloatArr(byte* prevA, void (Layer::* func)(float*))
	{
		float* _data = new float[TOTAL_PIXEL];
		for (int i = 0; i < TOTAL_PIXEL; i++) _data[i] = (float)prevA[i];
		(this->*func)(_data);
		delete[] _data;
	}
	void WeightOffset(float* prevA, int num, float dy)
	{
		int offset = 0;
		float wArr[9];
		for (int i = 0; i < TOTAL_PIXEL; i += offset)
		{
			for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) wArr[j * 3 + k] = L_RATE * prevA[i + j * COLUMNS + k] * dy;
			for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) weight[num][i + j * COLUMNS + k] += ((j == 1 && k == 1) ? 1.0 : 0.1) * (((double)wArr[j * 3 + k] + (double)wArr[4]) / 2.0);
			if (i == TOTAL_PIXEL - COLUMNS * 2 - 3) break;
			else offset = (i % COLUMNS == COLUMNS - 3) ? 3 : 1;
		}
	}
	float* answers, ** weight, * bias;
} inputLayer(128, TOTAL_PIXEL, true, false, ActiveFunction::relu), outputLayer(10, 10, false, true, ActiveFunction::softmax);
int main()
{
	srand((unsigned int)time(NULL)); rand();
	Data::ResetData(true);
	_FOPEN
	for (int i = 0; i < 60000; i++)
	{
		Data::ReadNextTrain();
		inputLayer.Forward(Data::image);
		outputLayer.Forward(inputLayer.activation);

		outputLayer.SetAnswer(Data::label);
		outputLayer.Backward(inputLayer.activation);

		inputLayer.SetAnswer(Data::label);
		inputLayer.Backward(Data::image);

		//outputLayer.FindAnswer(Data::label);
	}
	Data::ResetData(false);
	float cnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		Data::ReadNextTest();
		inputLayer.Forward(Data::image);
		outputLayer.Forward(inputLayer.activation);
		outputLayer.FindAnswer(Data::label, cnt);
	}
	printf("¥¿½T²v =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	_FCLOSE;
	return 0;
}
