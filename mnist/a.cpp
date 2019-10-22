/*
��@�Ʀr�޿�P�_
*/
#ifdef D



#include "mnist.h"

#include "stdlib.h"
#include "time.h"
#include "math.h"

FILE * f = fopen("D:\\Desktop\\mnist\\a.txt", "w");

// �ͦ��H���v���Pbias
void InitNeural(float& b, float* ww)
{
	for (int i = 0; i < TOTAL_PIXEL; i++) ww[i] = rand() * 1.0 / RAND_MAX;		//  0.0 .. 1.0
	b = rand() * 1.0 / RAND_MAX - 0.5;	// -0.5 .. 0.5
}

// �޿�^�k(���v��b+Sigma(���x*�q���v��w))
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
	float  ans = 0;	// �зǸѵ�
	float* ww = new float[TOTAL_PIXEL];	// �w���v���A�C�ӹ����@��(784)
	float  yy = 0;	// �w�����G
	float  b = 0;	// bias
	InitNeural(b, ww);	// random bias, weight
	float  r = 0.0001;	// �ǲ߳t�v


	float cnt = 0;
	float data_cnt = 0;
	for (int item = 0; item < NUMBER_OF_ITEMS; item++)
	{
		byte* data = Data::imageList[item];
		byte  trueAnswer = Data::labelList[item];
		if (trueAnswer == 0)
		{
			G(data, ww, b, yy);

			float dy = 1 - yy;  // �����q��
			//float dy = 0 - yy;  // �����S�q��
			for (int j = 0; j < TOTAL_PIXEL; j++) ww[j] = ww[j] + r * data[j] * dy * (yy * (1 - yy));
			b = b + r * dy * (yy * (1 - yy));
				
			if (yy > 0.5) cnt++; // �����q��
			//if (yy < 0.5) cnt++; // �����S�q��
			data_cnt++;

			fprintf(f, "ans = %d  dy = %2.6f  guess = %f  ", trueAnswer, dy, yy);
			fprintf(f, "b offset = %f\n", r * dy * (yy * (1 - yy)));
		}
	}
	printf("���T�v = %f%%  %.f / %.f\n", (cnt / data_cnt) * 100.0, cnt, data_cnt);
	printf("b = %f\n\n", b);

	delete[] ww;
	fclose(f);
	return 0;
}

/*
�Ϥ���ƿ�J = data
10��bias -> �q���ͦ��H���Ʀr��վ�
784���v��ww -> �q���ͦ��H���Ʀr��վ�
10�ӵ��Gyy -> { �C�ӵ��G = ���v+sigma���*�v�� }

�P�{��ans[10]���
�վ� ww[784], b[10]
*/

#endif