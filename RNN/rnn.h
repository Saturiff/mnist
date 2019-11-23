#pragma once
#include <iostream>
#pragma warning(disable:4996) // secure
#define INPUT_SIZE        7
#define TRAIN_ITEMS       60000
#define TEST_ITEMS        10000
#define TRAIN_INPUT_FILE  "D:/Desktop/neural network/RNN/rnn-train-input-v2"
#define TRAIN_ANSWER_FILE "D:/Desktop/neural network/RNN/rnn-train-answer-v2"
#define TEST_INPUT_FILE   "D:/Desktop/neural network/RNN/rnn-test-input-v2"
#define TEST_ANSWER_FILE  "D:/Desktop/neural network/RNN/rnn-test-answer-v2"
typedef unsigned char byte;

class Data
{
public:
	static byte  ans;
	static byte* input;
	static void ResetData();
	static void NextTrainAnswer();
	static void NextTestAnswer();
	static void ReadNextTrain();
	static void ReadNextTest();
private:
	Data();
	~Data();
	static void ReadNextAnswer(const char[]);
	static void ReadNextInput(const char[]);
	static int offsetOfAnswer;
	static int offsetOfInput;
};
