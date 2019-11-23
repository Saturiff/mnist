#include "rnn.h"
#include "mnist_debug.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE            0.1
#define ACTIONS_PRE_DATA  7
#define STATUS_PRE_ACTION 7
#define SEQ_ACTION        7
enum class ActiveFunction { sigmoid, relu, softmax };
class Layer
{
public:
	Layer(Layer* prevLayer, int neuronNum, int inputPreNeuron, bool isInputLayer, bool isOutputLayer, bool isEnableBias, ActiveFunction activeMode)
	{
		this->prevLayer = prevLayer, this->neuronNum = neuronNum, this->inputPreNeuron = inputPreNeuron, this->isInputLayer = isInputLayer, this->isOutputLayer = isOutputLayer, this->isEnableBias = isEnableBias, this->activeMode = activeMode, answers = new float[neuronNum], weight = new float* [neuronNum], bias = new float[neuronNum], activation = new float[neuronNum], z = new float[neuronNum];
		for (int i = 0; i < neuronNum; i++) weight[i] = new float[inputPreNeuron];
		InitLayer();
	}
	~Layer()
	{
		delete[] answers, weight, bias, activation;
	}
	void InitLayer()
	{
		for (int i = 0; i < neuronNum; i++)
		{
			z[i] = activation[i] = answers[i];
			bias[i] = (isEnableBias) ? rand() * 1.0 / RAND_MAX - 0.5 : 1.0;
			for (int j = 0; j < inputPreNeuron; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
		}
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
			for (int j = 0; j < inputPreNeuron; j++) z[i] += (double)weight[i][j] * (double)prevA[j] * 0.001;
			if      (activeMode == ActiveFunction::relu)    activation[i] = ReLU(z[i]);
			else if (activeMode == ActiveFunction::sigmoid) activation[i] = Sigmoid(z[i]);
		}
		if (activeMode == ActiveFunction::softmax) Softmax(z, activation, neuronNum);
	}
	void SetAnswer(byte answer)
	{
		for (int i = 0; i < neuronNum; i++) answers[i] = 0; answers[answer] = 1;
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
			bias[i] += (isEnableBias) ? L_RATE * dy : 0.0;
		}
	}
	void FindAnswer(byte trueAnswer)
	{
		int guessAnswer = 0;
		for (int i = 0; i < neuronNum; i++) if (activation[i] > activation[guessAnswer]) guessAnswer = i;
		fprintf(f, "activation = ");
		for (int i = 0; i < neuronNum; i++) fprintf(f, "%f ", activation[i]);
		fprintf(f, "====== guess = %d   answer = %d\n", guessAnswer, trueAnswer);
	}
	void FindAnswer(byte trueAnswer, float& cnt)
	{
		int guessAnswer = 0;
		for (int i = 0; i < neuronNum; i++) if (activation[i] > activation[guessAnswer]) guessAnswer = i;
		if (guessAnswer == trueAnswer) cnt++;
		fprintf(f, "activation = ");
		for (int i = 0; i < neuronNum; i++) fprintf(f, "%f ", activation[i]);
		fprintf(f, "====== guess = %d   answer = %d\n", guessAnswer, trueAnswer);
		//printf("activation = ");
		//for (int i = 0; i < neuronNum; i++) printf("%f ", activation[i]);
		//printf("\n====== guess = %d   answer = %d\n", guessAnswer, trueAnswer);
	}
	float* activation;
	int neuronNum;
protected:
	void ByteArr2FloatArr(byte* prevA, void (Layer::* func)(float*))
	{
		float* _data = new float[7];
		for (int i = 0; i < 7; i++) _data[i] = (float)prevA[i];
		(this->*func)(_data);
		delete[] _data;
	}
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
	float* answers, ** weight, * bias, * z;
	Layer* prevLayer;
	int inputPreNeuron;
	bool isInputLayer, isOutputLayer, isEnableBias;
	ActiveFunction activeMode;
} inputLayer(nullptr, 128, 7, true, false, false, ActiveFunction::relu), outputLayer(&inputLayer, 7, 7, false, true, false, ActiveFunction::softmax);
int main()
{
	srand(time(NULL)); rand();
	//float* mem1 = new float[inputLayer.neuronNum], * mem2 = new float[inputLayer.neuronNum];
	//for (int i = 0; i < inputLayer.neuronNum; i++) mem1[i] = 0;
	//for (int i = 0; i < inputLayer.neuronNum; i++) mem2[i] = 0;
	_FOPEN;
	Data::ResetData();
	for (int i = 0; i < TRAIN_ITEMS; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			Data::ReadNextTrain();
			Data::NextTrainAnswer();
			inputLayer.Forward(Data::input);
			outputLayer.Forward(inputLayer.activation);
			//for (int k = 0; k < inputLayer.neuronNum; k++) mem1[k] = inputLayer.activation[k];
			//for (int k = 0; k < inputLayer.neuronNum; k++) mem2[k] = mem1[k] + inputLayer.activation[k];
			//outputLayer.Forward(mem2);
			//for (int k = 0; k < outputLayer.neuronNum; k++) mem2[k] += outputLayer.activation[k];
			outputLayer.SetAnswer(Data::ans);
			outputLayer.Backward(inputLayer.activation);
			inputLayer.Backward(Data::input);
			//outputLayer.FindAnswer(Data::ans);
		}
		//for (int i = 0; i < inputLayer.neuronNum; i++) mem1[i] = 0;
		//for (int i = 0; i < inputLayer.neuronNum; i++) mem2[i] = 0;
	}
	Data::ResetData();
	float cnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		float localCnt = 0;
		for (int j = 0; j < 7; j++)
		{
			Data::ReadNextTest();
			Data::NextTestAnswer();
			inputLayer.Forward(Data::input);
			//for (int k = 0; k < inputLayer.neuronNum; k++) mem1[k] = inputLayer.activation[k];
			//for (int k = 0; k < inputLayer.neuronNum; k++) mem2[k] = mem1[k] + inputLayer.activation[k];
			//outputLayer.Forward(mem2);
			outputLayer.Forward(inputLayer.activation);
			outputLayer.FindAnswer(Data::ans, localCnt);
		}
		//for (int i = 0; i < inputLayer.neuronNum; i++) mem1[i] = 0;
		//for (int i = 0; i < inputLayer.neuronNum; i++) mem2[i] = 0;
		if (localCnt == 7) cnt += 1;
	}
	printf("¥¿½T²v =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	_FCLOSE;
}
