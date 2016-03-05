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
std::vector<double> desiredOutput;

double learningRate = 0.5;	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings& s){
   this->settings = s;
   cout << "settings set\n";
}

//----randomize weights-----
double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void randomizeWeightsInputHidden(std::vector<vector<double> > array){
	for(int i = 0; i < nInputNodes-10;i++){
		for(int j = 0; j < nHiddenNodes;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
		//set 10 bias nodes to zero
	for(int i = nInputNodes-10; i < nInputNodes;i++){
		for(int j = 0; j < nHiddenNodes;j++){
			array[i][j] = 0;
		}
	}
}

void randomizeWeightsHiddenOutput(std::vector<vector<double> > array){
	for(int i = 0; i < nHiddenNodes;i++){
		for(int j = 0; j < nOutputNodes;j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
}

void setDesiredOutput(std::vector<double> desiredOutput)){
	//get the label from the patch,
	//When you ask for getLabel from a feature you get a number of length 10
	//What do these numbers mean?
	
	//return an array with length 10 with all zero's and 1,1 
	}
//-------randomize weights end------


//------start FEEDFORWARD--------
double activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}
void feedforward(){
	double summedActivation = 0;

	for(int i = 0; i<nHiddenNodes;i++){
		for(int j=0;j<nInputNodes-1;j++){	
			summedActivation = summedActivation + input[j]*weightsInputHidden[j][i];
		}
		//use bias
		//????When the desired output is 0 there is no inluence of the bias????? NO IDEA IF THIS IS RIGHT
		for(int k=0;k<nOutputNodes;k++){
			summedActivation += weightsInputHidden[nInputNodes-1][i] * desiredOutput[k];
		}	
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
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}
	//This function only works for one output node.
double errorFunction(){
	double error = 0;
	for (int i=0; i< nOutputNodes;i++){
		error += (desiredOutput[i] - actualOutput[i])*(desiredOutput[i] - actualOutput[i]);
	}
	error *= 0.5;
	return error;
}

	//adjust weights with gradient decent, also only for one output node.
void adjustWeightsOutputUnits(){
	double deltaI = 0;
	for(int i = 0; i < nOutputNodes;i++){
		deltaI = (desiredOutput[i] - actualOutput[i]) * derivativeActivationFunction(actualOutput[i]);
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
			deltaO = (desiredOutput[j] - actualOutput[j]) * derivativeActivationFunction(actualOutput[j]);
			sumDeltaOWeights += deltaO * weightsHiddenOutput[i][j];
		}
		deltaI = derivativeActivationFunction(hiddenActivation[i]) * sumDeltaOWeights;
		sumDeltaOWeights = 0;
		
		for(int k = 0; k < nInputNodes-1; k++){
			weightsInputHidden[k][i] += learningRate * deltaI * input[k];
		}
		//adjust bias
		//????When the desired output is 0 there is no learning?????
		for(int l = 0; l < nOutputNodes; l++){
			weightsInputHidden[nInputNodes-1][i] += learningRate * deltaI * desiredOutput[l];
		}
	}
}

void backpropgation(){
	adjustWeightsHiddenUnit();
	adjustWeightsOutputUnits();
}
//--------end BACKPROPAGATION----


void MLPerceptron::train(vector<Feature>& randomFeatures){
	//nInputNodes 		= randomFeatures.at(0).size; This happens in the settingsfile
	//nHiddenNodes 		= (nInputNodes + nOutputNodes)/2;
	
	// set size of weight matrixes (in this case vectors of vectors)
	weightsHiddenOutput = vector<vector<double> >(nHiddenNodes, std::vector<double>(nOutputNodes,0.0));
	weightsInputHidden 	= vector<vector<double> >(nInputNodes, std::vector<double>(nHiddenNodes,0.0));
	
	//set size of input-, hiddenActivation-, and actalOutput-vectors
	input 				= vector<double>(nInputNodes,0.0);
	hiddenActivation 	= vector<double>(nHiddenNodes,0.0);
	actualOutput 		= vector<double>(nOutputNodes,0.0);
	desiredOutput 		= vector<double>(nOutputNodes,0.0);
	
	double error = 0.0;
	
	randomizeWeightsHiddenOutput(weightsHiddenOutput);
	randomizeWeightsInputHidden(weightsInputHidden);
	
	for(unsigned int i = 0; i < randomFeatures.size();i++){
		input.swap(randomFeatures.at(i).content);
		desiredOutput = setDesiredOutput()
		for(int j = 1; j < 10; j++){
			desiredOutput[j] = 0.0; //have to be changed according to the lable
		}
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
