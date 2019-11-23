#pragma once
#include <iostream>

template <typename T>
void Print1DArray(const char* arrName, T* arr, int size)
{
	printf("%s = ", arrName);
	for (int i = 0; i < size; i++) std::cout << arr[i] << ", ";
	printf("\n");
}

template <typename T>
void Print2DArray(const char* arrName, T** arr, int sizeX, int sizeY)
{
	printf("%s =\n", arrName);
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++) std::cout << arr[i][j] << ", ";
		printf("\n");
	}
	printf("\n");
}
