//#define MAIN
#ifdef MAIN

#include <iostream>
#include "stdlib.h"
#include "time.h"
#include "math.h"
#pragma warning(disable:6031) // rand
#define INPUT_SIZE 7
#define OUTPUT_SIZE 10
#define POOL_SIZE 10
#define ITEM_PRICE 5
int r() { return round(rand() * 1.0 / RAND_MAX * 7); }

void Forward()
{

}
void Backward()
{

}
int main()
{
	srand(time(NULL)); rand();
	int** input = new int* [INPUT_SIZE], * ansOutput = new int[OUTPUT_SIZE];
	for (int i = 0; i < INPUT_SIZE; i++) input[i] = new int[INPUT_SIZE];

	

	delete[] input, ansOutput;
	return 0;
}
#endif
