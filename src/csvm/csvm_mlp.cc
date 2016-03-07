#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

//Neuron
//input connections, output connections, threshold, activation function, type
//desired output
//backprogation algorithm
//input vector

using namespace std;
using namespace csvm;

//-------start variables-------

std::vector<vector<double> > weightsHiddenOutput;
std::vector<vector<double> > weightsInputHidden;

std::vector<double> input;
std::vector<double> hiddenActivation;
std::vector<double> actualOutput;
std::vector<double> desiredOutput = {1,0,0,0,0,0,0,0,0,0};

int amountOfBiasNodes = 10;

unsigned int nOutputUnits;
unsigned int nHiddenUnits;
unsigned int nInputUnits;

double learningRate = 0.5;	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings& s){
   this->settings = s;
   cout << "settings set\n";
}

//----randomize weights-----
double MLPerceptron::fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void MLPerceptron::randomizeWeightsInputHidden(std::vector<vector<double> > array){
	for(int i = 0; i < nInputUnits-10;i++){
		for(int j = 0; j < nHiddenUnits;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
		//set 10 bias nodes to zero
	for(int i = nInputUnits-10; i < nInputUnits;i++){
		for(int j = 0; j < nHiddenUnits;j++){
			array[i][j] = 0;
		}
	}
}

void MLPerceptron::randomizeWeightsHiddenOutput(std::vector<vector<double> > array){
	for(int i = 0; i < nHiddenUnits;i++){
		for(int j = 0; j < nOutputUnits;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
}

void MLPerceptron::setDesiredOutput(Feature f){
	int label = f.getLabel();
	//get the label from the patch,
	//When you ask for getLabel from a feature you get a number of length 10
	//What do these numbers mean?
	
	//return an array with length 10 with all zero's and 1,1 
	}
//-------randomize weights end------


//------start FEEDFORWARD--------
double MLPerceptron::activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}
void MLPerceptron::feedforward(){
	double summedActivation = 0;

	for(int i = 0; i<nHiddenUnits;i++){
		for(int j=0;j<nInputUnits-amountOfBiasNodes;j++){	
			summedActivation += input[j]*weightsInputHidden[j][i];
		}
		//use bias
		//????When the desired output is 0 there is no inluence of the bias????? NO IDEA IF THIS IS RIGHT
		for(int i = 0;i<amountOfBiasNodes;i++){
			for(int j = 0;j<nHiddenUnits;j++){
				summedActivation += weightsInputHidden[nInputUnits-amountOfBiasNodes][j] * desiredOutput[i];
			}
		}
		hiddenActivation[i] = activationFunction(summedActivation);
		summedActivation = 0;
	}
	
	for(int i = 0; i<nOutputUnits;i++){
		for(int j=0;j<nHiddenUnits;j++){	
			summedActivation += hiddenActivation[j]*weightsHiddenOutput[j][i];
		}
		//biasHidden *= 
		//summedActivation += bias;
		actualOutput[i] = activationFunction(summedActivation);
		summedActivation = 0;
	}
}
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double MLPerceptron::derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}
	//This function only works for one output node.
double MLPerceptron::errorFunction(){
	double error = 0;
	for (int i=0; i< nOutputUnits;i++){
		error += (desiredOutput[i] - actualOutput[i])*(desiredOutput[i] - actualOutput[i]);
	}
	error *= 0.5;
	return error;
}

	//adjust weights with gradient decent, also only for one output node.
void MLPerceptron::adjustWeightsOutputUnits(){
	double deltaI = 0;
	for(int i = 0; i < nOutputUnits;i++){
		deltaI = (desiredOutput[i] - actualOutput[i]) * derivativeActivationFunction(actualOutput[i]);
		for(int j = 0; j < nHiddenUnits; j++){
			weightsHiddenOutput[j][i] += learningRate * deltaI * hiddenActivation[j];
		}
	}
}

void MLPerceptron::adjustWeightsHiddenUnit(){
	double deltaO = 0;
	double deltaI = 0;
	double sumDeltaOWeights = 0;
	
	for(int i = 0; i < nHiddenUnits;i++){
		for(int j = 0; j < nOutputUnits; j++){
			deltaO = (desiredOutput[j] - actualOutput[j]) * derivativeActivationFunction(actualOutput[j]);
			sumDeltaOWeights += deltaO * weightsHiddenOutput[i][j];
		}
		deltaI = derivativeActivationFunction(hiddenActivation[i]) * sumDeltaOWeights;
		sumDeltaOWeights = 0;
		
		for(int j = 0; j < nInputUnits-amountOfBiasNodes; j++){
			weightsInputHidden[j][i] += learningRate * deltaI * input[j];
		}
		//adjust bias
		//????When the desired output is 0 there is no learning?????
		for(int j = 0; j < amountOfBiasNodes; j++){
			weightsInputHidden[nInputUnits-(amountOfBiasNodes - j)][i] += learningRate * deltaI * desiredOutput[j];
		}
	}
}

void MLPerceptron::backpropgation(){
	adjustWeightsHiddenUnit();
	adjustWeightsOutputUnits();
}
//--------end BACKPROPAGATION----


void MLPerceptron::train(vector<Feature>& randomFeatures){
	//nInputUnits 		= randomFeatures.at(0).size; This happens in the settingsfile
	//nHiddenUnits 		= (nInputUnits + nOutputUnits)/2;
	
	// set size of weight matrixes (in this case vectors of vectors)
	weightsHiddenOutput = vector<vector<double> >(nHiddenUnits, std::vector<double>(nOutputUnits,0.0));
	weightsInputHidden 	= vector<vector<double> >(nInputUnits, std::vector<double>(nHiddenUnits,0.0));
	
	//set size of input-, hiddenActivation-, and actalOutput-vectors
	input 				= vector<double>(nInputUnits,0.0);
	hiddenActivation 	= vector<double>(nHiddenUnits,0.0);
	actualOutput 		= vector<double>(nOutputUnits,0.0);
	desiredOutput 		= vector<double>(nOutputUnits,0.0);
	
	double error = 0.0;
	
	randomizeWeightsHiddenOutput(weightsHiddenOutput);
	randomizeWeightsInputHidden(weightsInputHidden);
	
	for(unsigned int i = 0; i < randomFeatures.size();i++){
		//input.swap(randomFeatures.at(i).content);
		//setDesiredOutput(randomFeatures.at(i));
		
		feedforward();
		error = errorFunction();
		std::cout << "error1: "  << error << std::endl;
		backpropgation();
		std::cout << "actualOutput[0]: "  << actualOutput[0] << std::endl;
		//std::cout << "desiredOutput[indexInput]: "  << desiredOutput << std::endl;
		std::cout << std::endl;
	}
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
