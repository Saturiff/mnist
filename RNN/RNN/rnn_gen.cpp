//#define GEN
#ifdef GEN

#include <iostream>
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031; disable:4996) // rand
#define INPUT_SIZE 7
#define TRAIN_QUANTITY 60000
#define TEST_QUANTITY 10000
#define TRAIN_INPUT_FILE "D:/Desktop/neural network/RNN/rnn-train-input"
#define TRAIN_ANSWER_FILE "D:/Desktop/neural network/RNN/rnn-train-answer"
#define TEST_INPUT_FILE "D:/Desktop/neural network/RNN/rnn-test-input"
#define TEST_ANSWER_FILE "D:/Desktop/neural network/RNN/rnn-test-answer"
typedef unsigned char byte;

int r() { return round(rand() * 1.0 / RAND_MAX * 6); }
void FormatArray(int* idx, byte** arr2d)
{
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++) arr2d[i][j] = 0;
		//printf("i = %d, j = %d\n", i, idx[i]);
		arr2d[i][idx[i]] = 1;
	}
}
void PrintRes(int* dataIdx, int* ansIdx, byte** data2d, byte** ans2d)
{
	for (int i = 0; i < INPUT_SIZE; i++) printf("%d ", dataIdx[i]); printf("\n");
	for (int i = 0; i < INPUT_SIZE; i++) printf("%d ", ansIdx[i]); printf("\n"); printf("\n");
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++) printf("%d ", data2d[i][j]);
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++) printf("%d ", ans2d[i][j]);
		printf("\n");
	}
}
void WriteToFile(const char* inputFilePath, const char* ansFilePath, int itemQuantity)
{
	int* dataIdx = new int[INPUT_SIZE], * ansIdx = new int[INPUT_SIZE];
	byte** data2d = new byte * [INPUT_SIZE], ** ans2d = new byte * [INPUT_SIZE];
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		data2d[i] = new byte[INPUT_SIZE];
		ans2d[i] = new byte[INPUT_SIZE];
	}
	FILE* fInput = fopen(inputFilePath, "wb"), * fAns = fopen(ansFilePath, "wb");
	for (int i = 0; i < itemQuantity; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			dataIdx[j] = r();
			if (dataIdx[j] == 0 || dataIdx[j] == 1 || dataIdx[j] == 2)      ansIdx[j] = 0;
			else if (dataIdx[j] == 3 || dataIdx[j] == 4 || dataIdx[j] == 5) ansIdx[j] = dataIdx[j] - 2;
			else if (dataIdx[j] == 6)                                       ansIdx[j] = 4;
		}
		FormatArray(dataIdx, data2d);
		FormatArray(ansIdx, ans2d);
		
		for (int i = 0; i < INPUT_SIZE; i++) for (int j = 0; j < INPUT_SIZE; j++)
		{
			fwrite(&data2d[i][j], sizeof(byte), 1, fInput);
			fwrite(&ans2d[i][j], sizeof(byte), 1, fAns);
		}
	}
	fclose(fInput);
	fclose(fAns);
}
void Gen()
{
	WriteToFile(TRAIN_INPUT_FILE, TRAIN_ANSWER_FILE, TRAIN_QUANTITY);
	WriteToFile(TEST_INPUT_FILE, TEST_ANSWER_FILE, TEST_QUANTITY);
}
int main()
{
	srand((unsigned int)time(NULL)); rand();
	Gen();
}
#endif
