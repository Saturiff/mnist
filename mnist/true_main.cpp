#include "mnist.h"

#include "stdlib.h"
#include "time.h"
#include "math.h"

#define QUANTITY_OF_NUMBER 10
FILE* f = fopen("D:\\Desktop\\mnist\\a.txt", "w");

// 生成隨機權重與bias
void InitNeural(float* b, float** ww)
{
	for (int i = 0; i < QUANTITY_OF_NUMBER; i++)
	{
		b[i] = rand() % 101 * 0.01 - 0.5;	// -0.5 .. 0.5
		for (int j = 0; j < NUMBER_OF_TOTAL_PIXEL; j++) ww[i][j] = rand() % 101 * 0.01;	// 0.0 .. 1.0
	}
}

void SetAnswer(float* ansArr, byte ans)
{
	for (int i = 0; i < QUANTITY_OF_NUMBER; i++) ansArr[i] = 0; ansArr[ans] = 1;
}

// 邏輯回歸(偏權值b+Sigma(資料x*電腦權重w))
void G(byte* data, float** w, float* b, float* yy)
{
	for (int num = 0; num < QUANTITY_OF_NUMBER; num++)
	{
		yy[num] = b[num];
		for (int pix = 0; pix < NUMBER_OF_TOTAL_PIXEL; pix++) yy[num] += w[num][pix] * (data[pix]/255) * 0.01; // max = 7.84 _ min = 0.0
		yy[num] /= 7.84;
		yy[num] = exp(yy[num]) / (exp(yy[num]) + 1);
	}
}

int main()
{
	srand((unsigned int)time(NULL)); rand();
	MnistMain();
	float*  ans  = new float [QUANTITY_OF_NUMBER];	// 標準解答，在正確的位置為1，其餘為0
	float** ww   = new float*[QUANTITY_OF_NUMBER];	// 預測權重，每種結果的每個像素各一個(7840 = 10*784)
	float*  yy   = new float [QUANTITY_OF_NUMBER];	// 預測結果，每種結果一個(10)
	float*  b    = new float [QUANTITY_OF_NUMBER];	// bias，每種結果一個(10)
	for (int i = 0; i < QUANTITY_OF_NUMBER; i++)
	{
		ans[i] = yy[i] = b[i] = 0;
		ww[i] = new float[NUMBER_OF_TOTAL_PIXEL]; 
		for (int j = 0; j < QUANTITY_OF_NUMBER; j++) ww[i][j] = 0;
	}
	InitNeural(b, ww);	// random bias, weight
	
	float  r = 0.01;	// 學習速率

	float cnt = 0, cntPart = 0;
	for (int item = 0; item < NUMBER_OF_ITEMS; item++)
	{
		byte* data       = Data::imageList[item];
		byte  trueAnswer = Data::labelList[item];
		SetAnswer(ans, trueAnswer);
		G(data, ww, b, yy);
		int guessAnswer = 0;
		for (int num = 0; num < QUANTITY_OF_NUMBER; num++)	// 由預測值陣列中的最大值當作此次解
		{
			float dy = ans[num] - yy[num];
			//if(trueAnswer == 1 && num == 1) fprintf(f, "dy = %f  ", dy);
			for (int j = 0; j < NUMBER_OF_TOTAL_PIXEL; j++) ww[num][j] += r * data[j] * dy * (yy[num] * (1 - yy[num]));
			b[num] += r * dy * (yy[num] * (1 - yy[num]));
			if (yy[num] > yy[guessAnswer]) guessAnswer = num;
		}
		if (guessAnswer == trueAnswer) cnt++;
		if (item > 50000 && guessAnswer == trueAnswer) cntPart++;
		//if (trueAnswer == 1) fprintf(f, "== yy == [%d] = %f  ans = %d guess = %d\n", 1, yy[1], trueAnswer, guessAnswer);
	}
	printf("總正確率\t= %.2f%%\t(%.f / %.f)\n", (cnt / (NUMBER_OF_ITEMS*1.0)) * 100.0, cnt, NUMBER_OF_ITEMS*1.0);
	printf("訓練後正確率\t= %.2f%%\t(%.f / %.f)\n", (cntPart / 10000.0) * 100.0, cntPart, 10000.0);

	delete[] ans, ww, yy;
	fclose(f);
	return 0;
}
