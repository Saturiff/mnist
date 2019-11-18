//#define GEN_NEW
#ifdef GEN_NEW
#include "rnn.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031; disable:4996) // rand

int r() { return round(rand() * 1.0 / RAND_MAX * 6); }
void FormatArray(int* idx, byte** arr2d)
{
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++) arr2d[i][j] = 0;
		arr2d[i][idx[i]] = 1;
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
	WriteToFile(TRAIN_INPUT_FILE, TRAIN_ANSWER_FILE, TRAIN_ITEMS);
	WriteToFile(TEST_INPUT_FILE, TEST_ANSWER_FILE, TEST_ITEMS);
}
int main()
{
	srand((unsigned int)time(NULL)); rand();
	Gen();
}
#endif
