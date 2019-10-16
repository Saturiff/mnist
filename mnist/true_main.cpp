#include "mnist.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:4244) // double -> float loss data
#pragma warning(disable:6031) // ignored rand
#define NUM_QUNATITY 10
#define L_RATE 0.01

void InitNeural(float* answerArray, float* guessArray, float* bias, float** weight)
{
	srand((unsigned int)time(NULL)); rand();
	for (int i = 0; i < NUM_QUNATITY; i++)
	{
		answerArray[i] = guessArray[i] = 0;
		weight[i] = new float[NUMBER_OF_TOTAL_PIXEL];
		bias[i] = rand() * 1.0 / RAND_MAX - 0.5;	// -0.5 .. 0.5
		for (int j = 0; j < NUMBER_OF_TOTAL_PIXEL; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;	// -0.5 .. 0.5
	}
}
void SetAnswer(float* ansArr, byte answerArray)
{
	for (int i = 0; i < NUM_QUNATITY; i++) ansArr[i] = 0; ansArr[answerArray] = 1;
}
void Forward(byte* data, float** w, float* bias, float* guessArray)
{
	for (int i = 0; i < NUM_QUNATITY; i++)
	{
		guessArray[i] = bias[i];
		for (int j = 0; j < NUMBER_OF_TOTAL_PIXEL; j++) guessArray[i] += w[i][j] * ((float)data[j]/255.0) * 0.001;
		guessArray[i] = exp(guessArray[i]) / (exp(guessArray[i]) + 1);
	}
}
void WeightOffset(int num, byte* data, float** w, float dy) // 將附近像素一同調整，由兩data平均影響 -> double layer
{
	int offset = 0;
	for (int i = 0; i < NUMBER_OF_TOTAL_PIXEL; i+=offset)
	{
		for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) w[num][i + j * NUMBER_OF_COLUMNS + k] += ((j == 1 && k == 1) ? 1.0 : 0.1) * L_RATE * (float)data[i + j * NUMBER_OF_COLUMNS + k] * dy;
		if(i % NUMBER_OF_COLUMNS == NUMBER_OF_COLUMNS - 3)
		{
			if (i == NUMBER_OF_TOTAL_PIXEL - NUMBER_OF_COLUMNS * 2 - 3) break;
			offset = 3;
		}
		else offset = 1;
	}
}
void Backward(byte* data, float* guessArray, float** w, float* bias, float* answerArray)
{
	for (int i = 0; i < NUM_QUNATITY; i++)
	{
		float dy = answerArray[i] - guessArray[i];
		WeightOffset(i, data, w, dy); // 90.33 43 s
		bias[i] += L_RATE * dy;
	}
}
int FindAnswer(float* guessArray, byte trueAnswer)
{
	int guessAnswer = 0, cnt = 0;
	for (int j = 0; j < NUM_QUNATITY; j++) if (guessArray[j] > guessArray[guessAnswer]) guessAnswer = j;
	if (guessAnswer == trueAnswer) cnt++;
	return cnt;
}
int main()
{
	float *answerArray = new float[NUM_QUNATITY], **weight = new float* [NUM_QUNATITY], *guessArray = new float[NUM_QUNATITY], *bias = new float[NUM_QUNATITY];
	InitNeural(answerArray, guessArray, bias, weight);
	byte* data = nullptr, trueAnswer = 0;
	Data::ResetData();
	for (int i = 0; i < TRAIN_ITEMS; i++)
	{
		Data::ReadNextTrain();
		data = Data::image;
		trueAnswer = Data::label[0];
		SetAnswer(answerArray, trueAnswer);
		Forward(data, weight, bias, guessArray);
		Backward(data, guessArray, weight, bias, answerArray);
	}
	Data::ResetData();
	float cnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		Data::ReadNextTest();
		data = Data::image;
		trueAnswer = Data::label[0];
		SetAnswer(answerArray, trueAnswer);
		Forward(data, weight, bias, guessArray);
		cnt += FindAnswer(guessArray, trueAnswer);
	}
	printf("正確率\t= %.2f%%\t(%.f / %.f)\n", (cnt / (TEST_ITEMS *1.0)) * 100.0, cnt, TEST_ITEMS *1.0);
	delete[] answerArray, weight, guessArray;
	return 0;
}
