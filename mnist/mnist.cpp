#include "mnist.h"

Data::Data() { }
Data::~Data() { }
byte  Data::label = 0;
byte* Data::image = 0;
int   Data::offsetOfLabel = 0;
int   Data::offsetOfImage = 0;
void  Data::ResetData()
{
	label = offsetOfImage = offsetOfLabel = 0;
	delete[] image; image = new byte[TOTAL_PIXEL];
}
void Data::ReadNextLabel(const char fileName[])
{
	FILE* labelFile = fopen(fileName, "rb");
		fseek(labelFile, offsetOfLabel + sizeof(int) * 2, 0);
		fread(&label, sizeof(byte), 1, labelFile);
	fclose(labelFile);
	offsetOfLabel++;
}
void Data::ReadNextImage(const char fileName[])
{
	FILE* imageFile = fopen(fileName, "rb");
		fseek(imageFile, offsetOfImage + sizeof(int) * 4, 0);
		fread(image, sizeof(byte), TOTAL_PIXEL, imageFile);
	fclose(imageFile);
	offsetOfImage += TOTAL_PIXEL;
}
void Data::ReadNextTrain()
{
	ReadNextLabel(TRAIN_LABELS_FILE);
	ReadNextImage(TRAIN_IMAGES_FILE);
}
void Data::ReadNextTest()
{
	ReadNextLabel(TEST_LABELS_FILE);
	ReadNextImage(TEST_IMAGES_FILE);
}
