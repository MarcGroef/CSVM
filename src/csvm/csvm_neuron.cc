#include <csvm/csvm_neuron.h>

using namespace std;
using namespace csvm;

double* inputWeights;
double* inputValue;
      
double* outputWeights;
double* outputValue;

double summedActivation;
double threshold;
      
//This value is in the formulate to calculate the summed input, but i don't know what it does
//It is neuron specific
double b;  
double activationValue;

void Neuron::Neuron(double* inWeights, double* inValues){
	this->inputWeights = inWeights;
	this->inputValue = inValues;
	outputWeights = randomizeWeights();
}
//-----start------ Handy functions
double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double* randomizeWeights(){
	double array[10];
	for(int i = 0; i<10;i++)
	{
		array[i] = fRand(-0.5,0.5);
	}
	return array;
}
//-----end----- Handy functions


	
	
