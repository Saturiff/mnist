#pragma once
#define TRAIN

#ifdef TRAIN
#define LABELS_FILE_NAME		"train-labels.idx1-ubyte"	// number
#define IMAGES_FILE_NAME		"train-images.idx3-ubyte"	// pixel
#define NUMBER_OF_ITEMS			60000
#elif TEST
#define LABELS_FILE_NAME		"t10k-labels.idx1-ubyte"
#define IMAGES_FILE_NAME		"t10k-images.idx3-ubyte"
#define NUMBER_OF_ITEMS			10000
#endif // TRAIN or TEST

#define NUMBER_OF_ROWS			28
#define NUMBER_OF_COLUMNS		28
#define NUMBER_OF_TOTAL_PIXEL	784 // rows * columns

#include <iostream>
typedef unsigned char byte;

// class for storage mnist file data
class Data
{
public:
	static byte*  labelList; // [item_index]-[label]
	static byte** imageList; // [item_index]-[image_pixel_array]

	friend void ReadLabelsFile(const char[]);
	friend void ReadImagesFile(const char[]);

private:
	Data();
	~Data();

	static void AddLabel(byte  inLabel);
	static void AddImage(byte* inPixel);

	static int indexOfLabel;
	static int indexOfImage;
};

// storage mnist file to Data, need about 6s
int MnistMain();
