#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

//-------------variables-----------------------
const int nInputNodes = 2 + 1;
const int nHiddenNodes = 2;
const int nOutputNodes = 1;
const int rowSize = nHiddenNodes;

double learningRate = 0.5;	

//double biasHidden = 0;

double weightsInputHidden[nInputNodes][nHiddenNodes];
double weightsHiddenOutput[nHiddenNodes][nOutputNodes];

//double input[nInputNodes];
double input[4][2]={{1,1},{1,0},{0,1},{0,0}};
double hiddenActivation[nHiddenNodes];
double actualOutput[nOutputNodes];
double desiredOutput[4] = {0,1,1,0};
int indexInput = 0;

//-----------endvariables------------------------	

double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void randomizeWeightsInputHidden(double array[][nHiddenNodes]){
	for(int i = 0; i < nInputNodes-1;i++){
		for(int j = 0; j < nHiddenNodes;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
	
	//set bias weight to zero
	for(int i = 0; i < nHiddenNodes;i++){
		array[nInputNodes-1][i] = 0;
	}
}

void randomizeWeightsHiddenOutput(double array[][nOutputNodes]){
	for(int i = 0; i < nHiddenNodes;i++){
		for(int j = 0; j < nOutputNodes;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
}


double activationFunction(double summedActivation){
	//std::cout << "1/(1+exp(-summedActivation)): "  << 1/(1+exp(-summedActivation)) << std::endl;
	return 1/(1+exp(-summedActivation));
}

double derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}
double sumUnits(int indexHidden,int size, double** activation, double** weights){
	double sumUnitsTimesWeights = 0;
	//double b = iets
	
	for(int i=0;i<nInputNodes;i++){	
		sumUnitsTimesWeights+=activation[indexInput][i]*weights[i][indexHidden];
	}
	//sumUnitsTimesWeights += b;
	return sumUnitsTimesWeights;
}
void feedforward(){
	double summedActivation = 0;

	for(int i = 0; i<nHiddenNodes;i++){
		for(int j=0;j<nInputNodes-1;j++){	
			summedActivation = summedActivation + input[indexInput][j]*weightsInputHidden[j][i];
		}
		//use bias
		summedActivation += weightsInputHidden[nInputNodes-1][i] * desiredOutput[indexInput];	
		hiddenActivation[i] = activationFunction(summedActivation);
		summedActivation = 0;
	}
	
	for(int i = 0; i<nOutputNodes;i++){
		for(int j=0;j<nHiddenNodes;j++){	
			summedActivation = summedActivation + hiddenActivation[j]*weightsHiddenOutput[j][i];
		}
		//biasHidden *= 
		//summedActivation += bias;
		actualOutput[i] = activationFunction(summedActivation);
		summedActivation = 0;
	}
}
//This function only works for one output node.
double errorFunction(){
	double error = 0;
	for (int i=0; i< nOutputNodes;i++){
		error += (desiredOutput[indexInput] - actualOutput[i])*(desiredOutput[indexInput] - actualOutput[i]);
	}
	error *= 0.5;
	return error;
}
//adjust weights with gradient decent, also only for one output node.
void adjustWeightsOutputUnits(){
	double deltaI = 0;
	for(int i = 0; i < nOutputNodes;i++){
		deltaI = (desiredOutput[indexInput] - actualOutput[i]) * derivativeActivationFunction(actualOutput[i]);
		for(int j = 0; j < nHiddenNodes; j++){
			weightsHiddenOutput[j][i] += learningRate * deltaI * hiddenActivation[j];
		}
	}
}

void adjustWeightsHiddenUnit(){
	double deltaO = 0;
	double deltaI = 0;
	double sumDeltaOWeights = 0;
	
	for(int i = 0; i < nHiddenNodes;i++){
		for(int j = 0; j < nOutputNodes; j++){
			deltaO = (desiredOutput[indexInput] - actualOutput[j]) * derivativeActivationFunction(actualOutput[j]);
			sumDeltaOWeights += deltaO * weightsHiddenOutput[i][j];
		}
		deltaI = derivativeActivationFunction(hiddenActivation[i]) * sumDeltaOWeights;
		sumDeltaOWeights = 0;
		
		for(int k = 0; k < nInputNodes-1; k++){
			weightsInputHidden[k][i] += learningRate * deltaI * input[indexInput][k];
		}
		//adjust bias
		weightsInputHidden[nInputNodes-1][i] += learningRate * deltaI * desiredOutput[indexInput];
	}
}

void backpropgation(){
	adjustWeightsHiddenUnit();
	adjustWeightsOutputUnits();
}
int main( void )
{
	double error;
	int epochs = 1000000;

	randomizeWeightsInputHidden(weightsInputHidden);
	randomizeWeightsHiddenOutput(weightsHiddenOutput);
	 for (int i = 0; i < nInputNodes; ++i)
    {
        for (int j = 0; j < nHiddenNodes; ++j)
        {
            std::cout << weightsInputHidden[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
     for (int i = 0; i < nHiddenNodes; ++i)
    {
        for (int j = 0; j < nOutputNodes; ++j)
        {
            std::cout  << weightsHiddenOutput[i][j] << ' ';
        }
        std::cout << std::endl;
    }
	
	
	//some epochs loop
	for(int i = 0; i < epochs;i++){
		indexInput = rand()%4;
		std::cout << "indexInput: "  << indexInput << std::endl;
		feedforward();
		error = errorFunction();
		std::cout << "error1: "  << error << std::endl;
		backpropgation();
		std::cout << "actualOutput[0]: "  << actualOutput[0] << std::endl;
		std::cout << "desiredOutput[indexInput]: "  << desiredOutput[indexInput] << std::endl;
		std::cout << std::endl;
	}
	
	
	
	 for (int i = 0; i < nInputNodes; ++i)
    {
        for (int j = 0; j < nHiddenNodes; ++j)
        {
            std::cout << weightsInputHidden[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    
     for (int i = 0; i < nHiddenNodes; ++i)
    {
        for (int j = 0; j < nOutputNodes; ++j)
        {
            std::cout << weightsHiddenOutput[i][j] << ' ';
        }
        std::cout << std::endl;
    }
	
	return 0;
}
