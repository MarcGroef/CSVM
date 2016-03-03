#include <stdlib.h>
#include <math.h>
	
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
double sumInputUnits(double* inputValue, double* inputWeights,int size){
	double sumInputUnitsTimesWeights = 0;
	double b = inputWeights[0];
	for(int i=0;i<size;i++)
	{	
		sumInputUnitsTimesWeights+=inputValue[i]*inputWeights[i];
	}
	return sumInputUnitsTimesWeights;
}

double activationFunction(double summedActivation){
	return 1/(1+exp (-summedActivation) );
}

int main( void )
{
const unsigned int nInputNodes = 10;
const unsigned int nHiddenNodes = 10;
const unsigned int nOutputNodes = 10;

double learningRate = 0.2;	
	
double weightsInputHidden[nInputNodes][nHiddenNodes];
double weightsHiddenOutput[nHiddenNodes][nOutputNodes];

double input[nInputNodes];
double hiddenActivation[nHiddenNodes];
double ActualOutput[nOutputNodes];
double desiredOutput[nOutputNodes] = {1,0,0,0,0,0,0,0,0,0};

double summedActivation = 0;

for(int i = 0; i<nInputNodes;i++){
	input[i] = (double)rand() / RAND_MAX;;
	}
	
randomizeWeights(nInputNodes, nHiddenNodes, weightsInputHidden);
randomizeWeights(nHiddenNodes, nOutputNodes, weightsHiddenOutput);

for(int i = 0; i < nHiddenNodes;i++){
	summedActivation += sumInputUnits(input,weightsInputHidden[i],nHiddenNodes);
	hiddenActivation[i] = activationFunction(summedActivation);
	}
}
