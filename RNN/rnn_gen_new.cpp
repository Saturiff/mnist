#include "rnn.h"
#ifdef GEN_NEW

#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031; disable:4996) // rand

int* dataIdx = new int[INPUT_SIZE], * ansIdx = new int[INPUT_SIZE];
byte** data2d = new byte * [INPUT_SIZE];
FILE* fInput, * fAns;
void Init()
{
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		dataIdx[i] = ansIdx[i] = 0;
		data2d[i] = new byte[INPUT_SIZE];
		for (int j = 0; j < INPUT_SIZE; j++) data2d[i][j] = 0;
	}
}
void WriteToFile(const char* inputFilePath, const char* ansFilePath, int itemQuantity)
{
	fInput = fopen(inputFilePath, "wb"), fAns = fopen(ansFilePath, "wb");
	for (int i = 0; i < itemQuantity; i++)
	{
		int money = 0;
		for (int j = 0; j < INPUT_SIZE; j++) // main logic loop: 7 action pre seq
		{
			// 0: idle
			// 1: can buy
			// 2: drop A
			// 3: drop B
			// 4: drop C
			dataIdx[j] = (int)round(rand() * 1.0 / RAND_MAX * 6);
			if (dataIdx[j] == 0 || dataIdx[j] == 1 || dataIdx[j] == 2) // in money 1, 5, 10
			{
				money += (dataIdx[j] == 0) ? 1 : (dataIdx[j] == 1) ? 5 : (dataIdx[j] == 2) ? 10 : 0;
				ansIdx[j] = 0; // idle
			}
			else if (dataIdx[j] == 3 || dataIdx[j] == 4 || dataIdx[j] == 5) // select item A, B, C
			{
				if (money >= 5) // can buy item now
				{
					ansIdx[j] = dataIdx[j] - 2;
					money = 0;
				}
				else ansIdx[j] = 0; // keep idle
			}
			else if (dataIdx[j] == 6) ansIdx[j] = money = 0; // return money -> idle
		}
		for (int i = 0; i < INPUT_SIZE; i++)
		{
			for (int j = 0; j < INPUT_SIZE; j++) data2d[i][j] = 0;
			data2d[i][dataIdx[i]] = 1;
		}
		for (int j = 0; j < 7; j++)
		{
			for (int k = 0; k < 7; k++) fwrite(&data2d[j][k], sizeof(byte), 1, fInput);
			fwrite(&ansIdx[j], sizeof(byte), 1, fAns);
		}
	}
	fclose(fInput);
	fclose(fAns);
}
void Gen()
{
	Init();
	WriteToFile(TRAIN_INPUT_FILE, TRAIN_ANSWER_FILE, TRAIN_ITEMS);
	WriteToFile(TEST_INPUT_FILE, TEST_ANSWER_FILE, TEST_ITEMS);
}
int main()
{
	srand((unsigned int)time(NULL)); rand();
	Gen();
}
#endif
