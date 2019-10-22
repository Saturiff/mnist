#pragma once
#include <iostream>
#pragma warning(disable:4996) // secure
#define TRAIN_LABELS_FILE		"train-labels.idx1-ubyte"
#define TRAIN_IMAGES_FILE		"train-images.idx3-ubyte"
#define TRAIN_ITEMS				60000
#define TEST_LABELS_FILE		"t10k-labels.idx1-ubyte"
#define TEST_IMAGES_FILE		"t10k-images.idx3-ubyte"
#define TEST_ITEMS				10000
#define ROWS			        28
#define COLUMNS		            28
#define TOTAL_PIXEL	            784
#define MAX_COLOR_VALUE			255.0
typedef unsigned char byte;

class Data
{
public:
	static byte  label;
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
