#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <string>

//Neuron
//input connections, output connections, threshold, activation function, type
//desired output
//backprogation algorithm
//input vector

using namespace std;
using namespace csvm;

//-------start variables-------

//These variables actually have to come from the settingsfile
//int nHiddenUnits = 59;
//int nInputUnits = 108 + 10;
//int nOutputUnits = 10;
std::vector<int> layerSizes;

std::vector<vector<double> > weightsHiddenOutput;
std::vector<vector<double> > weightsInputHidden;
std::vector<vector<vector<double> > > weights;

std::vector<double> input;
std::vector<double> hiddenActivation;
std::vector<double> actualOutput;
std::vector<vector<double> > layers;

std::vector<vector<double> > deltas;
std::vector<double> desiredOutput;

int amountOfBiasNodes;

double learningRate = 0.05;
	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings s){
   this->settings = s;
   cout << "settings set\n";
}

//----randomize weights-----
double MLPerceptron::fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void MLPerceptron::randomizeWeights(std::vector<vector<double> >& array){
	for(unsigned int i = 0; i < array.size();i++){
		for(unsigned int j = 0; j < array[0].size();j++){
			array[i][j] = fRand(-0.5,0.5);
			//std::cout << weights.at(0)[i][j] << std::endl;
		}
	}
}

void MLPerceptron::setDesiredOutput(Feature f){
	int label = f.getLabelId();
	//std::cout << label << std::endl;
	desiredOutput = vector<double>(settings.nOutputUnits,0.0);
	desiredOutput.at(label) = 1;
	}

//-------randomize weights end------


//------start FEEDFORWARD--------
double MLPerceptron::activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}

void MLPerceptron::calculateActivationLayer(int firstLayerSize ,int secondLayerSize, std::vector<double> &firstLayer, std::vector<vector<double> > weights,std::vector<double> &secondLayer){
	double summedActivation = 0;
	
	for(int i=0; i<secondLayerSize;i++){
		for(int j=0;j<firstLayerSize;j++){	
			summedActivation += firstLayer[j] * weights[j][i];
		}
		secondLayer[i] = activationFunction(summedActivation);
	}	
}
void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++){
		calculateActivationLayer(layerSizes[i],layerSizes[i+1],layers[i],weights[i],layers[i+1]);
		//std::cout << "in feedforward,deltas[settings.nLayers-1][i]"  << deltas[settings.nLayers-1][i]<< std::endl;
	}
}
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double MLPerceptron::derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}

double MLPerceptron::errorFunction(){
	double error = 0;
	
	for (int i=0; i< layerSizes[settings.nLayers-1];i++){
		error += (desiredOutput[i] - layers[settings.nLayers-1][i])*(desiredOutput[i] - layers[settings.nLayers-1][i]);
	}
	error *= 0.5;
	return error;
}

	//adjust weights with gradient decent, also only for one output node.

void MLPerceptron::calculateError(int index){
	//std::cout << "in calculateError, index: " << index << std::endl;
	if (index == 0){
		return;
	}
	if(index == settings.nLayers-1){
		outputDelta();
	//	std::cout << "in calculateError,outputDelta "  << std::endl;
	}
	if(index > 0 && index <= settings.nLayers-2){
		hiddenDelta(index);
	//	std::cout << "in calculateError,hiddenDelta "  << std::endl;
	}
	index--;
	return calculateError(index);
}

void MLPerceptron::outputDelta(){
	for(int i = 0; i < layerSizes[settings.nLayers-1];i++){
		deltas[settings.nLayers-1][i] = (desiredOutput[i] - layers[settings.nLayers-1][i])*derivativeActivationFunction(layers[settings.nLayers-1][i]);
		
		//std::cout << "in outputDelta,deltas[settings.nLayers-1][i]"  << deltas[settings.nLayers-1][i]<< std::endl;
		//std::cout << "in outputDelta,layers[settings.nLayers-1][i]"  << layers[settings.nLayers-1][i]<< std::endl;
	}
}
	
void MLPerceptron::hiddenDelta(int index){
	//std::cout << "in hiddenDelta,layerSizes[index]"  << layerSizes[index] << std::endl;
	//std::cout << "in hiddenDelta,layerSizes[index+1]"  << layerSizes[index+1] << std::endl;
	double sumDeltaWeights = 0;
	//loop over all hidden layer nodes
	for(int i = 0; i < layerSizes[index];i++){
		for(int j = 0; j < layerSizes[index+1];j++){
			sumDeltaWeights += deltas[index+1][j] * weights[index][i][j];
		}
		deltas[index][i] = sumDeltaWeights*derivativeActivationFunction(layers[index][i]);
		sumDeltaWeights = 0;
	}
}	

void MLPerceptron::adjustWeights(int index, int sizeLeftLayer, int sizeRightLayer){
	for(int i = 0; i < sizeLeftLayer; i++){
		for(int j = 0; j < sizeRightLayer; j++){
			weights[index][i][j] += learningRate * deltas[index+1][j] * layers[index][i];
//			std::cout << weights[index][i][j] << " ";
		}
//		std::cout << std::endl;
	}
//	std::cout << std::endl;
}


void MLPerceptron::backpropgation(){
	//std::cout << "in backpropgation " << std::endl;
	calculateError(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-2;i++){
		//std::cout << "in adjusting weights: " << std::endl;
		adjustWeights(i, layerSizes[i], layerSizes[i+1]);
	}
}
//--------end BACKPROPAGATION----

void MLPerceptron::initializeVectors(){
	int maxNumberOfNodes = 0;
	layerSizes		= vector<int>(settings.nLayers,0);
	
	layerSizes.at(0) = settings.nInputUnits;
	layerSizes.at(1) = settings.nHiddenUnits;
	layerSizes.at(2) = settings.nOutputUnits;
	
	//returns max layer size
	for(unsigned int i = 0; i < layerSizes.size();i++){
		if(maxNumberOfNodes < layerSizes[i]){
			maxNumberOfNodes = layerSizes[i];	
		}
	}
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);
	
	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));
	layers			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	for(int i = 0;i < settings.nLayers-1;i++){
		randomizeWeights(weights.at(i));
	}
	//set bias nodes
	amountOfBiasNodes = layerSizes.at(settings.nLayers-1);
}


void MLPerceptron::train(vector<Feature>& randomFeatures){
	double error = 0.0;
	
	initializeVectors();
	//std::cout << "in train, settings.inputLayers: " << settings.nLayers << std::endl;
	
	for(unsigned int i = 0; i < randomFeatures.size();i++){
		layers.at(0) = randomFeatures.at(i).content;
		setDesiredOutput(randomFeatures.at(i));
		feedforward();
		backpropgation();
		error = errorFunction();
		std::cout << "error: "  << error << std::endl;
		//std::cout << "actualOutput[0]: "  << actualOutput[0] << std::endl;
		//std::cout << "desiredOutput[indexInput]: "  << desiredOutput << std::endl;
		//std::cout << std::endl;

	}
	for(int i=0;i<10;i++){
		std::cout << "desiredOutput[1]: "  << desiredOutput[i] << std::endl;
		std::cout << "actualOutput[2][i]: "  << layers[2][i] << std::endl;
		}
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
