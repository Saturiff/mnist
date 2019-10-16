#pragma once
#include <iostream>
#define TRAIN_LABELS_FILE		"train-labels.idx1-ubyte"
#define TRAIN_IMAGES_FILE		"train-images.idx3-ubyte"
#define TRAIN_ITEMS				60000
#define TEST_LABELS_FILE		"t10k-labels.idx1-ubyte"
#define TEST_IMAGES_FILE		"t10k-images.idx3-ubyte"
#define TEST_ITEMS				10000
#define NUMBER_OF_ROWS			28
#define NUMBER_OF_COLUMNS		28
#define NUMBER_OF_TOTAL_PIXEL	784
typedef unsigned char byte;

class Data
{
public:
	static byte* label;
	static byte* image;
	static void ResetData();
	static void ReadNextTrain();
	static void ReadNextTest();
private:
	Data();
	~Data();
	static void ReadNextLabel(const char[]);
	static void ReadNextImage(const char[]);
	static int offsetOfLabel;
	static int offsetOfImage;
};
