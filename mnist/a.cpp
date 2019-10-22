/*
單一數字邏輯判斷
*/
#ifdef D



#include "mnist.h"

#include "stdlib.h"
#include "time.h"
#include "math.h"

FILE * f = fopen("D:\\Desktop\\mnist\\a.txt", "w");

// 生成隨機權重與bias
void InitNeural(float& b, float* ww)
{
	for (int i = 0; i < TOTAL_PIXEL; i++) ww[i] = rand() * 1.0 / RAND_MAX;		//  0.0 .. 1.0
	b = rand() * 1.0 / RAND_MAX - 0.5;	// -0.5 .. 0.5
}

// 邏輯回歸(偏權值b+Sigma(資料x*電腦權重w))
void G(byte x[], float w[], float b, float& yy)
{
	yy = b;
	for (int pix = 0; pix < TOTAL_PIXEL; pix++) yy += w[pix] * (x[pix] / 255) * 0.01; // max = 7.84 _ min = 0.0
	yy /= 7.84;
	printf("yy = %.2f\n", yy);
	yy = exp(yy) / (exp(yy) + 1);
}

int main()
{
	srand((unsigned int)time(NULL)); rand();
	MnistMain();
	float  ans = 0;	// 標準解答
	float* ww = new float[TOTAL_PIXEL];	// 預測權重，每個像素一個(784)
	float  yy = 0;	// 預測結果
	float  b = 0;	// bias
	InitNeural(b, ww);	// random bias, weight
	float  r = 0.0001;	// 學習速率


	float cnt = 0;
	float data_cnt = 0;
	for (int item = 0; item < NUMBER_OF_ITEMS; item++)
	{
		byte* data = Data::imageList[item];
		byte  trueAnswer = Data::labelList[item];
		if (trueAnswer == 0)
		{
			G(data, ww, b, yy);

			float dy = 1 - yy;  // 模擬猜中
			//float dy = 0 - yy;  // 模擬沒猜中
			for (int j = 0; j < TOTAL_PIXEL; j++) ww[j] = ww[j] + r * data[j] * dy * (yy * (1 - yy));
			b = b + r * dy * (yy * (1 - yy));
				
			if (yy > 0.5) cnt++; // 模擬猜中
			//if (yy < 0.5) cnt++; // 模擬沒猜中
			data_cnt++;

			fprintf(f, "ans = %d  dy = %2.6f  guess = %f  ", trueAnswer, dy, yy);
			fprintf(f, "b offset = %f\n", r * dy * (yy * (1 - yy)));
		}
	}
	printf("正確率 = %f%%  %.f / %.f\n", (cnt / data_cnt) * 100.0, cnt, data_cnt);
	printf("b = %f\n\n", b);

	delete[] ww;
	fclose(f);
	return 0;
}

/*
圖片資料輸入 = data
10個bias -> 電腦生成隨機數字後調整
784個權重ww -> 電腦生成隨機數字後調整
10個結果yy -> { 每個結果 = 偏權+sigma資料*權重 }

與現實ans[10]比對
調整 ww[784], b[10]
*/

#endif