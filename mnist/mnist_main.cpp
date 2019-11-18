#include "mnist.h"
#include "mnist_debug.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define NUMBER_QUNATITY 10
#define L_RATE 0.01
void InitNeural(float* answerArray, float* output, float* bias, float** weight)
{
	srand((unsigned int)time(NULL)); rand();
	for (int i = 0; i < NUMBER_QUNATITY; i++)
	{
		answerArray[i] = output[i] = 0;
		weight[i]      = new float[TOTAL_PIXEL];
		bias[i]        = rand() * 1.0 / RAND_MAX - 0.5;
		for (int j = 0; j < TOTAL_PIXEL; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
	}
}
void SetAnswer(byte answerArray, float* ansArr)
{
	for (int i = 0; i < NUMBER_QUNATITY; i++) ansArr[i] = 0; ansArr[answerArray] = 1;
}
void Forward(byte* data, float** weight, float* bias, float* guessArray)
{
	for (int i = 0; i < NUMBER_QUNATITY; i++)
	{
		guessArray[i] = bias[i];
		for (int j = 0; j < TOTAL_PIXEL; j++) guessArray[i] += weight[i][j] * ((float)data[j]/MAX_COLOR_VALUE) * 0.001;
		guessArray[i] = exp(guessArray[i]) / (exp(guessArray[i]) + 1);
	}
}
void WeightOffset(byte* data, int num, float** weight, float dy)
{
	int offset = 0;
	float wArr[9]; 
	for (int i = 0; i < TOTAL_PIXEL; i += offset)
	{
		for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) wArr [j * 3 + k] = L_RATE * (float)data[i + j * COLUMNS + k] * dy;
		for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) weight [num][i + j * COLUMNS + k] += ((j == 1 && k == 1) ? 1.0 : 0.1) * (((double)wArr[j * 3 + k] + (double)wArr[4]) / 2.0);
		if (i == TOTAL_PIXEL - COLUMNS * 2 - 3) break;
		else offset = (i % COLUMNS == COLUMNS - 3) ? 3 : 1;
	}
}
void Backward(byte* data, float** weight, float* bias, float* guessArray, float* answerArray)
{
	for (int i = 0; i < NUMBER_QUNATITY; i++)
	{
		float dy = answerArray[i] - guessArray[i];
		WeightOffset(data, i, weight, dy);
		bias[i] += L_RATE * dy;
	}
}
void FindAnswer(byte trueAnswer, float* guessArray, float& cnt)
{
	int guessAnswer = 0;
	for (int i = 0; i < NUMBER_QUNATITY; i++) if (guessArray[i] > guessArray[guessAnswer]) guessAnswer = i;
	if (guessAnswer == trueAnswer) cnt++;
}
int main()
{
	float *answerArray = new float[NUMBER_QUNATITY], **weight = new float*[NUMBER_QUNATITY], *bias = new float[NUMBER_QUNATITY], *output = new float[NUMBER_QUNATITY];
	InitNeural(answerArray, output, bias, weight);
	Data::ResetData();
	_FOPEN
	for (int i = 0; i < TRAIN_ITEMS; i++)
	{
		Data::ReadNextTrain();
		SetAnswer(Data::label, answerArray);
		Forward  (Data::image, weight, bias, output);
		Backward (Data::image, weight, bias, output, answerArray);
	}
	_FCLOSE
	Data::ResetData();
	float cnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		Data::ReadNextTest();
		Forward(Data::image, weight, bias, output);
		FindAnswer(Data::label, output, cnt);
	}
	printf("¥¿½T²v =\t%.2f%% (%.f / %.f)\n", (cnt / (TEST_ITEMS * 1.0)) * 100.0, cnt, TEST_ITEMS * 1.0);
	delete[] answerArray, weight, bias, output;
	return 0;
}
