//#define TEST
#ifdef TEST
#include "rnn.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE            0.001
#define ACTIONS_PRE_DATA  7
#define STATUS_PRE_ACTION 7
#define SEQ_ACTION        7
#define _FOPEN            f = fopen("D:/Desktop/neural network/RNN/o.txt", "w");
#define _FCLOSE           fclose(f);
#define _FPRINT(x)        fprintf(f, x);
FILE* f;
void InitNeural(float** weight, float* bias, float* guess, double* ans, float* mem)
{
	srand((unsigned int)time(NULL)); rand();
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		weight[i] = new float[7];
		guess[i] = ans[i] = mem[i] = 0;
		bias[i] = rand() * 1.0 / RAND_MAX - 0.5;
		for (int j = 0; j < 7; j++) weight[i][j] = rand() * 1.0 / RAND_MAX - 0.5;
	}
}
void SetAnswer(double* ansArr)
{
	for (int i = 0; i < 7; i++) ansArr[i] = 0; ansArr[Data::ans] = 1;
}
void Forward(float** weight, float* bias, float* guessArray, float* mem)
{
	for (int i = 0; i < 7; i++) // 神經元
	{
		guessArray[i] = bias[i];
		for (int j = 0; j < 7; j++) guessArray[i] += weight[i][j] * (double)Data::input[j] * 0.001; // 每個byte
		guessArray[i] = exp(guessArray[i]) / (exp(guessArray[i]) + 1) + mem[i];
	}
	for (int i = 0; i < 7; i++) mem[i] = guessArray[i];
}
void Backward(float** weight, float* bias, float* guessArray, double* answerArray)
{
	for (int i = 0; i < 7; i++)
	{
		float dy = answerArray[i] - guessArray[i];
		for (int j = 0; j < 7; j++) weight[i][j] += L_RATE * (double)Data::input[j] * dy;
		bias[i] += L_RATE * dy;
	}
}
void ResetRNNMemory(float* mem)
{
	for (int i = 0; i < 7; i++) mem[i] = 0;
}
void FindAnswer(float* guessArray, float& cnt)
{
	printf("============\n");
	printf("answer = %d\n", Data::ans);
	printf("guess = ");
	for (int i = 0; i < 7; i++)printf("%f ", guessArray[i]);
	printf("\n");
	int guessAnswer = 0;
	for (int i = 0; i < 7; i++) if (guessArray[i] > guessArray[guessAnswer]) guessAnswer = i;
	if (guessAnswer == Data::ans) cnt++;
	//printf("ans = %d  guess = %d\n", Data::ans, guessAnswer);
}
int main()
{
	_FOPEN
		float** weight = new float* [INPUT_SIZE], * bias = new float[INPUT_SIZE], * guess = new float[INPUT_SIZE], * mem = new float[INPUT_SIZE];
	double* answer = new double[INPUT_SIZE];
	InitNeural(weight, bias, guess, answer, mem);
	Data::ResetData();
	for (int i = 0; i < TRAIN_ITEMS; i++)
	{
		Data::ReadNextTrain();
		for (int j = 0; j < 7; j++)
		{
			Data::NextTrainAnswer();
			SetAnswer(answer);
			Forward(weight, bias, guess, mem);
			Backward(weight, bias, guess, answer);
		}
		ResetRNNMemory(mem);
	}
	/*
	printf("weight\n");
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 7; j++) printf("%lf ", weight[i][j]);
		printf("\n");
	}
	printf("\n");
	printf("bias\n");
	for (int i = 0; i < 7; i++) printf("%f ", bias[i]);
	printf("\n");
	*/
	Data::ResetData();
	float cnt = 0, gCnt = 0;
	for (int i = 0; i < TEST_ITEMS; i++)
	{
		Data::ReadNextTest();
		for (int j = 0; j < 7; j++)
		{
			Data::NextTestAnswer();
			Forward(weight, bias, guess, mem);
			FindAnswer(guess, cnt);
		}
		if (cnt == 7) gCnt++;
		cnt = 0;
		//printf("cnt = %.f\n", cnt);
		ResetRNNMemory(mem);
	}
	printf("正確率 =\t%.2f%% (%.f / %.f)\n", (gCnt / (TEST_ITEMS * 1.0)) * 100.0, gCnt, TEST_ITEMS * 1.0);
	delete[] answer, weight, bias, guess;
	_FCLOSE
		return 0;
}
#endif
