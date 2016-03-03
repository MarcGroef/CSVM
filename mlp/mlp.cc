#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

//-------------variables-----------------------
const int nInputNodes = 10;
const int nHiddenNodes = 10;
const int nOutputNodes = 10;

//double learningRate = 0.2;	
	
double weightsInputHidden[nInputNodes][nHiddenNodes];
double weightsHiddenOutput[nHiddenNodes][nOutputNodes];

double input[nInputNodes];
double hiddenActivation[nHiddenNodes];
double actualOutput[nOutputNodes];
double desiredOutput[nOutputNodes] = {1,0,0,0,0,0,0,0,0,0};

//-----------endvariables------------------------	

double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void randomizeWeights(int sizeRow,int sizeCollum, double array[10][10]){
	for(int i = 0; i < sizeRow;i++){
		for(int j = 0; j < sizeCollum;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
}
double sumUnits(double* inputValue, double* inputWeights,int size){
	double sumUnitsTimesWeights = 0;
	double b = inputWeights[0];
	for(int i=0;i<size;i++)
	{	
		sumUnitsTimesWeights+=inputValue[i]*inputWeights[i];
	}
	return sumUnitsTimesWeights + b;
}

double activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}

double derivativeActivationFunction(double summedActivation){
	return exp(summedActivation)/((exp(summedActivation) + 1)*(exp(summedActivation) + 1))
	}

void feedforward(){
	double summedActivation = 0;

	for(int i = 0; i < nHiddenNodes;i++){
	summedActivation += sumUnits(input,weightsInputHidden[i],nHiddenNodes);
	hiddenActivation[i] = activationFunction(summedActivation);
	}
	
for(int i = 0; i < nOutputNodes;i++){
	summedActivation += sumUnits(hiddenActivation,weightsHiddenOutput[i],nOutputNodes);
	actualOutput[i] = activationFunction(summedActivation);
	}
	
}

double errorFunction(){
	double error = 0;
	for (int i=0; i< nOutputNodes;i++){
		error += (desiredOutput[i] - actualOutput[i])*(desiredOutput[i] - actualOutput[i]);
	}
	error *= 0.5;
	return error;
}
//adjust weights with gradient decent
void gradientDecent(double error){
	deltaI = 0;
	for(int i = 0; i < nOutputNodes;i++){
		deltaI = 
		
		}
	
}

int main( void )
{
double error;
for(int i = 0; i<nInputNodes;i++){
	input[i] = (double)rand() / RAND_MAX;;
}
	
randomizeWeights(nInputNodes, nHiddenNodes, weightsInputHidden);
randomizeWeights(nHiddenNodes, nOutputNodes, weightsHiddenOutput);

feedforward();

error = errorFunction();

std::cout << error;
return 0;
}
