#include "rnn.h"

Data::Data() { }
Data::~Data() { }
byte  Data::ans = 0;
byte* Data::input = 0;
int   Data::offsetOfAnswer = 0;
int   Data::offsetOfInput = 0;
void  Data::ResetData()
{
	ans = offsetOfInput = offsetOfAnswer = 0;
	delete[] input; input = new byte[INPUT_SIZE];
}
void Data::NextTrainAnswer()
{
	ReadNextAnswer(TRAIN_ANSWER_FILE);
}
void Data::NextTestAnswer()
{
	ReadNextAnswer(TEST_ANSWER_FILE);
}
void Data::ReadNextAnswer(const char fileName[])
{
	FILE* labelFile = fopen(fileName, "rb");
	fseek(labelFile, offsetOfAnswer, 0);
	fread(&ans, sizeof(byte), 1, labelFile);
	fclose(labelFile);
	offsetOfAnswer++;
}
void Data::ReadNextInput(const char fileName[])
{
	FILE* imageFile = fopen(fileName, "rb");
	fseek(imageFile, offsetOfInput, 0);
	fread(input, sizeof(byte), INPUT_SIZE, imageFile);
	fclose(imageFile);
	offsetOfInput += INPUT_SIZE;
}
void Data::ReadNextTrain()
{
	ReadNextInput(TRAIN_INPUT_FILE);
}
void Data::ReadNextTest()
{
	ReadNextInput(TEST_INPUT_FILE);
}
