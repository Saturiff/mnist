#include "mnist.h"

Data::Data() { }
Data::~Data() { }
byte  Data::label = 0;
byte* Data::image = 0;
int   Data::offsetOfTrainLabel = 0;
int   Data::offsetOfTrainImage = 0;
int   Data::offsetOfTestLabel = 0;
int   Data::offsetOfTestImage = 0;
void  Data::ResetData()
{
	label =  offsetOfTrainLabel = offsetOfTestLabel = offsetOfTrainImage = offsetOfTestImage = 0;	
	delete[] image; image = new byte[TOTAL_PIXEL];
}
void Data::ReadNextLabel(const char fileName[], bool isTrain)
{
	FILE* labelFile = fopen(fileName, "rb");
	fseek(labelFile, ((isTrain) ? offsetOfTrainLabel : offsetOfTestLabel) + sizeof(int) * 2, 0);
	fread(&label, sizeof(byte), 1, labelFile);
	fclose(labelFile);
	(isTrain) ? offsetOfTrainLabel++ : offsetOfTestLabel++;
}
void Data::ReadNextImage(const char fileName[], bool isTrain)
{
	FILE* imageFile = fopen(fileName, "rb");
	fseek(imageFile, ((isTrain) ? offsetOfTrainImage : offsetOfTestImage) + sizeof(int) * 4, 0);
	fread(image, sizeof(byte), TOTAL_PIXEL, imageFile);
	fclose(imageFile);
	((isTrain) ? offsetOfTrainImage : offsetOfTestImage) += TOTAL_PIXEL;
}
void Data::ReadNextTrain()
{
	ReadNextLabel(TRAIN_LABELS_FILE, true);
	ReadNextImage(TRAIN_IMAGES_FILE, true);
}
void Data::ReadNextTest()
{
	ReadNextLabel(TEST_LABELS_FILE, false);
	ReadNextImage(TEST_IMAGES_FILE, false);
}
