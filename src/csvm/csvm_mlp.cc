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

std::vector<double> desiredOutput;

int amountOfBiasNodes = 0;


double learningRate = 0.1;	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings s){
   this->settings = s;
   cout << "settings set\n";
   cout << "Checking settingsvalues:\nnOutputUnits = " << settings.nOutputUnits << "\nnHiddenUnits = " << settings.nHiddenUnits << "\nnInputUnits = " << settings.nInputUnits << endl;
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
			std::cout << weights.at(0)[i][j] << std::endl;
		}
	}
}

*/
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

void MLPerceptron::calculateActivationLayer(int firstLayerSize ,int secondLayerSize, std::vector<double> firstLayer, std::vector<vector<double> > weights,std::vector<double> secondLayer){
	double summedActivation = 0;
	
	for(int i=0; i<secondLayerSize;i++){
		for(int j=0;j<firstLayerSize-amountOfBiasNodes;j++){	
			summedActivation += firstLayer[j] * weights[j][i];
		}
		//use bias
		//I believe this part is skipped if amountOfBiasNodes = 0		
		
		//for(int j = 0;m<amountOfBiasNodes;m++){
		//	for(int k = 0;n<settings.nHiddenUnits;n++){
		//		summedActivation += weightsInputHidden[settings.nInputUnits-amountOfBiasNodes][k] * desiredOutput[j];
		//	}
		//}
		
		secondLayer[i] = activationFunction(summedActivation);
	}	
}
void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++){
		calculateActivationLayer(layerSizes[i],layerSizes[i+1],layers[i],weights[i],layers[i+1]);
		//calculateActivationLayer(settings.nHiddenUnits,settings.nOutputUnits,hiddenActivation,weightsHiddenOutput,actualOutput);
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
	
	for (int i=0; i< settings.nOutputUnits;i++){
		error += (desiredOutput[i] - actualOutput[i])*(desiredOutput[i] - actualOutput[i]);
	}
	error *= 0.5;
	return error;
}

	//adjust weights with gradient decent, also only for one output node.
void MLPerceptron::adjustWeightsOutputUnits(){
	double deltaI = 0;

	for(int i = 0; i < settings.nOutputUnits;i++){
		deltaI = (desiredOutput[i] - actualOutput[i]) * derivativeActivationFunction(actualOutput[i]);
		for(int j = 0; j < settings.nHiddenUnits; j++){
			weightsHiddenOutput[j][i] += learningRate * deltaI * hiddenActivation[j];
		}
	}
}

void MLPerceptron::adjustWeightsHiddenUnit(){
	double deltaO = 0;
	double deltaI = 0;
	double sumDeltaOWeights = 0;
	
	for(int i = 0; i < settings.nHiddenUnits;i++){
		for(int j = 0; j < settings.nOutputUnits; j++){
			deltaO = (desiredOutput[j] - actualOutput[j]) * derivativeActivationFunction(actualOutput[j]);
			sumDeltaOWeights += deltaO * weightsHiddenOutput[i][j];
		}
		deltaI = derivativeActivationFunction(hiddenActivation[i]) * sumDeltaOWeights;
		sumDeltaOWeights = 0;
		
		for(int j = 0; j < settings.nInputUnits-amountOfBiasNodes; j++){
			weightsInputHidden[j][i] += learningRate * deltaI * input[j];
		}
		//adjust bias
		//????When the desired output is 0 there is no learning?????
		for(int j = 0; j < amountOfBiasNodes; j++){
			weightsInputHidden[settings.nInputUnits-(amountOfBiasNodes - j)][i] += learningRate * deltaI * desiredOutput[j];
		}
	}
}

void MLPerceptron::backpropgation(){
	adjustWeightsHiddenUnit();
	adjustWeightsOutputUnits();
	
	/* best thing would be if it works like this:
	for(int i = 0; i < numberOfLayers;i++){
	* backprop[i];
	* }
	I don't know if this is possible
	*/
}
//--------end BACKPROPAGATION----

void MLPerceptron::initializeVectors(){
	// set size of weight matrixes (in this case vectors of vectors)
	weightsHiddenOutput = vector<vector<double> >(settings.nHiddenUnits, std::vector<double>(settings.nOutputUnits,0.0));
	weightsInputHidden 	= vector<vector<double> >(settings.nInputUnits, std::vector<double>(settings.nHiddenUnits,0.0));
	
	//set size of input-, hiddenActivation-, and actalOutput-vectors
	input 				= vector<double>(settings.nInputUnits,0.0);
	hiddenActivation 	= vector<double>(settings.nHiddenUnits,0.0);
	actualOutput 		= vector<double>(settings.nOutputUnits,0.0);
	
	desiredOutput 		= vector<double>(settings.nOutputUnits,0.0);
	
	layerSizes			= vector<int>(settings.nLayers,0);
	weights				= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(settings.nInputUnits, std::vector<double>(settings.nOutputUnits,0.0)));
	layers				= vector<vector<double> >(settings.nLayers,std::vector<double>(settings.nInputUnits,0.0));
	
	layerSizes.at(0) = settings.nInputUnits;
	layerSizes.at(1) = settings.nHiddenUnits;
	layerSizes.at(2) = settings.nOutputUnits;
	
	//weights.at(0) = weightsInputHidden;
	//weights.at(1) = weightsHiddenOutput;
	
	//layers.at(0) = input;
	//layers.at(1) = hiddenActivation;
	//layers.at(2) = actualOutput;
	for(int i = 0;i < settings.layerSizes-1;i++){
		randomizeWeights(weights.at(i));
	}
}

void MLPerceptron::train(vector<Feature>& randomFeatures){
	//double error = 0.0;
	
	initializeVectors();
	
/*	
	for(unsigned int i = 0; i < randomFeatures.size();i++){
		input = randomFeatures.at(i).content;
		
		//for(int j=0; j<settings.nInputUnits; j++){
		//  std::cout << input.at(j) << std::endl;
		//}
		
		setDesiredOutput(randomFeatures.at(i));
		feedforward();
		error = errorFunction();
		std::cout << "error1: "  << error << std::endl;
		backpropgation();
		std::cout << "actualOutput[0]: "  << actualOutput[0] << std::endl;
		//std::cout << "desiredOutput[indexInput]: "  << desiredOutput << std::endl;
		std::cout << std::endl;
		std::cout << "randomFeature size: " << randomFeatures.size() << std::endl;

	}
	*/
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
