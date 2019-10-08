#ifdef NEURAL

/*
���g���� - ��@���g��
�Ҥl�V�m -> �վ��v�� -> ��ֻ~�t
���*�v�� -> ??��� -> ����{�ꪺ�ѵ�
x = data	���
w = weight	�v��
b = bias	���v�ȡA��ƨ�����{���I�O���Ī�
f			�E�y�B�ಾ���
*/

#include <iostream>
#include "stdlib.h"
#include "time.h"
#include "math.h"

// �ͦ��Ѹ�ƻP�з��v���o�X����
// ���ҡG���Z�`�M�j��60�h�^��1�A�_�h�^��0
// Sigma���T���*�v����A�A�ϥΥ��T�P�_���^�ǥ��T����
// (��Ƶ���, ��ư}�C, �з��v���}�C)
float f(int n, float x[], float w[])
{
	float sum = 0;
	for (int i = 0; i < n; i++) sum += w[i] * x[i];
	if (sum >= 60) return 1.0;
	return 0.0;
}

// �q���ǲߨ��
// ���v��b+Sigma���x*�q���v��w�A�^���޿�^�k���
// (��Ƶ���, ��ư}�C, �q���v���}�C, ���v��)
float g(int n, float x[], float w[], float b)
{
	float y = b;
	for (int i = 0; i < n; i++) y += w[i] * x[i] * 0.01; // Sigma
	printf("y = %.2f\n", y);
	return exp(y) / (exp(y) + 1.0);
}

// (��Ƶ���, ��ư}�C)
void set_example(int n, float x[])
{
	for (int i = 0; i < n; i++) x[i] = rand() % 100 + 1.0;
}


#define GEN_TIMES 500000.0

int main1()
{
	srand(time(NULL)); rand();
	
	float r = 0.00001;						// �ǲ߳t�v
	float w[3] = { 0.3, 0.3, 0.4 };			// �з��v��
	float ww[3] = { 0.1, 0.5, 0.4 };		// �q���v��
	float x[3];								// ���
	float b = -0.6;							// ���v��
	int m = 0;								// �֭p����
	for (int n = 0; n < GEN_TIMES; n++)
	{
		set_example(3, x);					// �ͦ��H�����
		float y = f(3, x, w);				// �зǪ���
		float yy = g(3, x, ww, b);			// �q������
		//printf("yy = %.2f\n", yy);
		float dy = y - yy;					// �зǻP�q�����t�Z
		for (int i = 0; i < 3; i++) ww[i] = ww[i] + r * x[i] * dy * (yy * (1 - yy)); // Sigma : �ۥ[�Ҧ� -> �v��+�ǲ߳t�v*���׮t�Z*(�q���ѵ�*(1-�q���ѵ�)
		b = b + r * dy * (yy * (1 - yy));	// ���v�Ƚվ� : ��l���v��+�ǲ߳t�v*���׮t�Z*(�q���ѵ�*(1-�q���ѵ�)
		//for (int i = 0; i < 3; i++) printf("%.2f\t", x[i]);	
		//printf("yy = %.2f y = %.2f\n", yy, y);
		if (n > GEN_TIMES / 2)				// �b�h���ǲ߫�}�l�p��
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
