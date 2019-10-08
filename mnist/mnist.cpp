#include "mnist.h"

Data::Data() { }
Data::~Data() { delete[] labelList, imageList; }
int    Data::indexOfLabel = 0;
int    Data::indexOfImage = 0;
byte*  Data::labelList    = new byte  [NUMBER_OF_ITEMS];
byte** Data::imageList    = new byte* [NUMBER_OF_ITEMS];
void   Data::AddLabel(byte  inLabel)  { labelList[indexOfLabel++] = inLabel; }
void   Data::AddImage(byte* inPixel) 
{
	imageList[indexOfImage] = new byte[NUMBER_OF_TOTAL_PIXEL];
	for (int i = 0; i < NUMBER_OF_TOTAL_PIXEL; i++) imageList[indexOfImage][i] = inPixel[i];
	indexOfImage++;
}

void NextInt32_Simple(FILE* filePtr)
{
	byte buffer[sizeof(int)] = { 0 };
	fread(buffer, sizeof(byte), sizeof(int), filePtr);
}

int NextInt32(FILE* filePtr)
{
	byte buffer[sizeof(int)] = { 0 };
	fread(buffer, sizeof(byte), sizeof(int), filePtr);
	int num = 0;
	for (int i = 0; i < sizeof(int); i++) num |= (int)buffer[i] << (sizeof(int) - i - 1) * 8; // bit offset 0, 8, 16, 24
	return num;
}

byte NextByte(FILE* filePtr)
{
	byte buffer[1] = { 0 };
	fread(buffer, sizeof(byte), 1, filePtr);
	return buffer[0];
}

void ReadLabelsFile(const char fileName[])
{
	FILE* labelFile;
	labelFile = fopen(fileName, "rb");
		for (int i = 0; i < 2; i++) NextInt32_Simple(labelFile); // pass useless data
		for (int i = 0; i < NUMBER_OF_ITEMS; i++) Data::AddLabel(NextByte(labelFile));
	fclose(labelFile);
}

void ReadImagesFile(const char fileName[])
{
	FILE* imageFile;
	imageFile = fopen(fileName, "rb");
		for (int i = 0; i < 4; i++) NextInt32_Simple(imageFile); // pass useless data
		byte* image = new byte[NUMBER_OF_TOTAL_PIXEL];
		for (int i = 0; i < NUMBER_OF_ITEMS; i++)
		{
			for (int j = 0; j < NUMBER_OF_TOTAL_PIXEL; j++) image[j] = NextByte(imageFile);
			Data::AddImage(image);
		}
	fclose(imageFile);
}

int MnistMain()
{
	ReadLabelsFile(LABELS_FILE_NAME);
	ReadImagesFile(IMAGES_FILE_NAME);

	return 0;
}

/*
Label file:
	int[2]	magic number(0x0081), numbers of items
	byte	label ..

Image file:
	int[4]	magic number(0x0083), numbers of images, number of rows, number of columns
	byte	pixel ..

magic number:
	0x00, 0x00, 0x08 -> unsigned byte, dimensions of the vector/matrix
*/
