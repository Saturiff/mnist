#ifdef NEURAL

/*
神經網路 - 單一神經元
例子訓練 -> 調整權重 -> 減少誤差
資料*權重 -> ??函數 -> 接近現實的解答
x = data	資料
w = weight	權重
b = bias	偏權值，資料到哪個臨界點是有效的
f			激勵、轉移函數
*/

#include <iostream>
#include "stdlib.h"
#include "time.h"
#include "math.h"

// 生成由資料與標準權重得出的解
// 情境：成績總和大於60則回傳1，否則回傳0
// Sigma正確資料*權重後，再使用正確判斷式回傳正確的解
// (資料筆數, 資料陣列, 標準權重陣列)
float f(int n, float x[], float w[])
{
	float sum = 0;
	for (int i = 0; i < n; i++) sum += w[i] * x[i];
	if (sum >= 60) return 1.0;
	return 0.0;
}

// 電腦學習函數
// 偏權值b+Sigma資料x*電腦權重w，回傳邏輯回歸函數
// (資料筆數, 資料陣列, 電腦權重陣列, 偏權值)
float g(int n, float x[], float w[], float b)
{
	float y = b;
	for (int i = 0; i < n; i++) y += w[i] * x[i] * 0.01; // Sigma
	printf("y = %.2f\n", y);
	return exp(y) / (exp(y) + 1.0);
}

// (資料筆數, 資料陣列)
void set_example(int n, float x[])
{
	for (int i = 0; i < n; i++) x[i] = rand() % 100 + 1.0;
}


#define GEN_TIMES 500000.0

int main1()
{
	srand(time(NULL)); rand();
	
	float r = 0.00001;						// 學習速率
	float w[3] = { 0.3, 0.3, 0.4 };			// 標準權重
	float ww[3] = { 0.1, 0.5, 0.4 };		// 電腦權重
	float x[3];								// 資料
	float b = -0.6;							// 偏權值
	int m = 0;								// 累計次數
	for (int n = 0; n < GEN_TIMES; n++)
	{
		set_example(3, x);					// 生成隨機資料
		float y = f(3, x, w);				// 標準的解
		float yy = g(3, x, ww, b);			// 電腦的解
		//printf("yy = %.2f\n", yy);
		float dy = y - yy;					// 標準與電腦的差距
		for (int i = 0; i < 3; i++) ww[i] = ww[i] + r * x[i] * dy * (yy * (1 - yy)); // Sigma : 相加所有 -> 權重+學習速率*答案差距*(電腦解答*(1-電腦解答)
		b = b + r * dy * (yy * (1 - yy));	// 偏權值調整 : 原始偏權值+學習速率*答案差距*(電腦解答*(1-電腦解答)
		//for (int i = 0; i < 3; i++) printf("%.2f\t", x[i]);	
		//printf("yy = %.2f y = %.2f\n", yy, y);
		if (n > GEN_TIMES / 2)				// 在多次學習後開始計數
		{
			if (y > 0.5 && yy < 0.5)		m++;
			else if (y < 0.5 && yy > 0.5)	m++;
		}
		//for (int i = 0; i < 3; i++) printf("%.2f\t", ww[i]); printf("\n");
	}
	//printf("m = %d\n", m);
	printf("m = %.2f%%\n", (m / GEN_TIMES) * 100.0);

	return 0;
}


#endif
