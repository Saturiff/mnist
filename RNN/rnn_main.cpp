#define MAIN
#ifdef MAIN
#include "rnn.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define L_RATE            0.00001
#define ACTIONS_PRE_DATA  7
#define STATUS_PRE_ACTION 7
#define SEQ_ACTION        7
#define _FOPEN            f = fopen("D:/Desktop/neural network/RNN/o.txt", "w");
#define _FCLOSE           fclose(f);
#define _FPRINT(x)        fprintf(f, x);
FILE* f;
void _inline Softmax(float* x, int size)
{
	float sum = 0.0;
	for (int i = 0; i < size; i++) sum += exp(x[i]);
	for (int i = 0; i < size; i++) x[i] = exp(x[i]) / sum;
}
int main()
{
	int size = 7;
	float* x = new float[size];
	x[0] = 1;
	x[1] = 2;
	x[2] = 3;
	x[3] = 4;
	x[4] = 1;
	x[5] = 2;
	x[6] = 3;
	Softmax(x, size);
	printf("Softmax = ");
	for (int i = 0; i < size; i++) printf("%f ", x[i]);
	printf("\n");
	return 0;
}
#endif
