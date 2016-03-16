#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <string>

//In here still needs to be the addition of bias nodes.
//Still unclear is how many bias nodes are needed and if all they layers except the output layer need a bias nodes.
//The bias nodes on the input layer learn according to the output
//But bias nodes in the 1,2 or 3 hidden layer, where do they learn according to?

//I will try to implement the bias nodes for the input layer. I will make the amount of bias nodes that can be added
//dynamic so we can always set the amount of nodes back to zero if it does not work.
using namespace std;
using namespace csvm;


/*
 * All the variables below are declared globally. They are used by all the methods in this class.
 * In the class there is a mix between calling methods with parameters and without.
 * This could lead to confussion.
 */
//-------start variables-------
std::vector<int> layerSizes;
std::vector<int> amountOfBiasNodesLayers;

std::vector<vector<vector<double> > > weights;
std::vector<vector<double> > layers;

std::vector<vector<double> > deltas;
std::vector<double> desiredOutput; 

double learningRate = 0.3;
	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings s){
   this->settings = s;
   cout << "settings set\n";
}

double MLPerceptron::fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void MLPerceptron::randomizeWeights(std::vector<vector<double> >& array,int indexLeftLayer){
	std::cout << std::endl;
	//std::cout << "Random weights, index: " << indexLeftLayer << std::endl;
	for(unsigned int i = 0; i < layerSizes[indexLeftLayer]-amountOfBiasNodesLayers[indexLeftLayer];i++){
		for(unsigned int j = 0; j < layerSizes[indexLeftLayer+1]-amountOfBiasNodesLayers[indexLeftLayer+1];j++){
			array[i][j] = fRand(-0.5,0.5);
			std::cout << array[i][j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;
}

void MLPerceptron::setDesiredOutput(Feature f){
	int label = f.getLabelId();
	//std::cout << label << std::endl;
	desiredOutput = vector<double>(settings.nOutputUnits,0.0);
	desiredOutput.at(label) = 1;
}
	
double MLPerceptron::errorFunction(){
	double error = 0;
	for (int i=0; i< layerSizes[settings.nLayers-1];i++){
		error += (desiredOutput[i] - layers[settings.nLayers-1][i])*(desiredOutput[i] - layers[settings.nLayers-1][i]);
		//std::cout << "desiredOutput[i] - layers[settings.nLayers-1][i]: " << desiredOutput[i] - layers[settings.nLayers-1][i] << std::endl;
	}
	error *= 0.5;
	
	return error;
}


//------start FEEDFORWARD--------

double MLPerceptron::activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}

void MLPerceptron::calculateActivationLayer(int leftLayerSize ,int rightLayerSize, std::vector<double> &leftLayer,std::vector<double> &rightLayer, std::vector<vector<double> > weights,int leftLayerIndex){
	double summedActivation = 0;
	//std::cout << "in calculateActivationLayer,leftlayerIndex: " << leftLayerIndex << std::endl;
	int sizeBiasNodesLeftLayer = amountOfBiasNodesLayers[leftLayerIndex];
	
	for(int i=0; i<rightLayerSize-amountOfBiasNodesLayers[leftLayerIndex+1];i++){
		for(int j=0;j<leftLayerSize-sizeBiasNodesLeftLayer;j++){	
				summedActivation += leftLayer[j] * weights[j][i];
				//std::cout << "in calculateActivationLayer,leftLayer[j]: " << leftLayer[j] << std::endl;
				//std::cout << "in calculateActivationLayer,weights[j][i]: " << weights[j][i] << std::endl;
			}
		//std::cout << "in calculateActivationLayer,summedActivation: " << summedActivation << std::endl;

	    for(int j = leftLayerSize-sizeBiasNodesLeftLayer; j < leftLayerSize;j++){
			//TODO
			//std::cout << "in calculateActivationLayer,leftlayerIndex , j: " << leftLayerIndex << j << std::endl;	
			summedActivation += weights[j][i] * desiredOutput[0];
		} 
		//std::cout << "in calculateActivationLayer,summedActivation+bias: " << summedActivation << std::endl;
		rightLayer[i] = activationFunction(summedActivation);
		//std::cout << "in calculateActivationLayer,summedActivation after activation function: " << rightLayer[i] << std::endl;
	}	
}

void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++){
		calculateActivationLayer(layerSizes[i],layerSizes[i+1],layers[i],layers[i+1],weights[i],i);
		//std::cout << "in feedforward,deltas[settings.nLayers-1][i]"  << deltas[settings.nLayers-1][i]<< std::endl;
	}
		//std::cout << "in feedforward, lekker: " << std::endl;		

}
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double MLPerceptron::derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}

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
		//std::cout << "in outputDelta,deltas["<< settings.nLayers-1 << "]["<< i << "]: " << deltas[settings.nLayers-1][i] << std::endl;

	}
}
	
void MLPerceptron::hiddenDelta(int index){
	//std::cout << "in hiddenDelta,layerSizes[index]"  << layerSizes[index] << std::endl;
	double sumDeltaWeights = 0;
	//loop over all hidden layer nodes
	for(int i = 0; i < layerSizes[index];i++){
		for(int j = 0; j < layerSizes[index+1];j++){
			sumDeltaWeights += deltas[index+1][j] * weights[index][i][j];
			//std::cout << "in calculateActivationLayer,deltas[index+1][j]: " << deltas[index+1][j] << std::endl;
			//std::cout << "in calculateActivationLayer,weights[j][i]: " << weights[index][i][j] << std::endl;
		}
		//std::cout << "in hiddenDelta,sumDeltaWeights " << sumDeltaWeights << std::endl;
		deltas[index][i] = sumDeltaWeights*derivativeActivationFunction(layers[index][i]);
		//std::cout << "in hiddenDelta,deltas["<< index << "]["<< i << "]: " << deltas[index][i] << std::endl;
		sumDeltaWeights = 0;
	}
	std::cout << std::endl;
}	

void MLPerceptron::adjustWeights(int index, int sizeLeftLayer, int sizeRightLayer){
	double a_j = 0;
	//std::cout << "in adjustWeights, weights[index][j][i]: " << index <<std::endl;
	for(int i = 0; i < sizeRightLayer ; i++){
		for(int j = 0; j < sizeLeftLayer; j++){
			if(j < sizeLeftLayer-amountOfBiasNodesLayers[index]){
				a_j = layers[index][j];  
			}
			else{
				a_j = desiredOutput[0];	
			}
			weights[index][j][i] += learningRate * deltas[index+1][i] * a_j;
			std::cout << weights[index][j][i] << " ";
		}
		std::cout << std::endl;
	}
	/*
	//Adjusting the bias weights:
	for(int i = 0; i < sizeRightLayer; i++){
		for(int j = sizeLeftLayer-amountOfBiasNodesLayers[index]; j < sizeLeftLayer; j++){
			weights[index][j][i] += learningRate * deltas[index+1][i] * desiredOutput[0];
			std::cout << weights[index][j][i] << " ";
		}
		std::cout << std::endl;
	}
//	std::cout << std::endl;
* */
}


void MLPerceptron::backpropgation(){
	calculateError(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-1;i++){
		//std::cout << "in adjusting weights: " << std::endl;
		adjustWeights(i, layerSizes[i], layerSizes[i+1]);
	}
}
//--------end BACKPROPAGATION----

void MLPerceptron::initializeVectors(){
	int maxNumberOfNodes = 0;
	layerSizes				= vector<int>(settings.nLayers,0);
	amountOfBiasNodesLayers	= vector<int>(settings.nLayers,0);
	
	layerSizes.at(0) = settings.nInputUnits;
	layerSizes.at(1) = settings.nHiddenUnits;
	layerSizes.at(2) = settings.nOutputUnits;
	
	amountOfBiasNodesLayers.at(0) = 1;
	amountOfBiasNodesLayers.at(1) = 0;
	amountOfBiasNodesLayers.at(2) = 0;
	
	for (int i = 0;i<settings.nLayers;i++){
		layerSizes[i] += amountOfBiasNodesLayers[i];
	}
	
	//returns max layer size
	for(unsigned int i = 0; i < layerSizes.size();i++){
		if(maxNumberOfNodes < layerSizes[i]){
			maxNumberOfNodes = layerSizes[i];	
		}
	}
	//Kind of a 'omslachtige' way to make the vectors. There should be a more aligant solution for this
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);

	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));
	layers			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	for(int i = 0;i < settings.nLayers-1;i++){
		randomizeWeights(weights[i],i);
	}
}

void MLPerceptron::train(vector<Feature>& randomFeatures){
	double error = 0.0;
	
	initializeVectors();
	std::cout << "in train, settings.inputLayers: " << settings.nLayers << std::endl;
	
	//Testing MLP with XOR
	std::vector<double> possibleOutput = vector<double>(4,0.0);
	std::vector<vector<double> > input = vector<vector<double> >(4,std::vector<double>(2,0.0));
	
	input[0][0] = 1;
	input[0][1] = 1;
	
	input[1][0] = 1;
	input[1][1] = 0;
	
	input[2][0] = 0;
	input[2][1] = 1;
	
	input[3][0] = 0;
	input[3][1] = 0;
	
	possibleOutput[0] = 0;
	possibleOutput[1] = 1;
	possibleOutput[2] = 1;
	possibleOutput[3] = 0;
	
	for(unsigned int i = 0; i < 10000;i++){ //randomFeatures.size()
		//layers.at(0) = randomFeatures.at(i).content;
		//setDesiredOutput(randomFeatures.at(i));
		
		//testing MLP with XOR
		int num = rand() % 4;
		layers[0] = input[num];
		desiredOutput[0] = possibleOutput[num];
		
		std::cout << "in train, desiredOutput[0]: " << desiredOutput[0] << std::endl;
		std::cout << "in train, layers[0][0]: " << layers[0][0] << std::endl;
		std::cout << "in train, layers[0][1]: " << layers[0][1] << std::endl;
		
		feedforward();
		backpropgation();
		error = errorFunction();
		
		std::cout << "error: "  << error << std::endl;
		
		for(int i=0;i<1;i++){
		std::cout << "desiredOutput[0]: "  << desiredOutput[i] << std::endl;
		std::cout << "actualOutput[2][i]: "  << layers[2][i] << std::endl;
	}
		//std::cout << std::endl;
	}
	
	
	// printing the weights for all weight matrixs
	for(int i = 0; i < settings.nLayers-1;i++){
		std::cout << "weight matrix: " << i << std::endl;
		for(int j = 0; j < layerSizes[i]; j++){
			for(int k = 0; k < layerSizes[i+1]; k++){
				std::cout << weights[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
}
	
	// printing the desired output against the actual ouput after the last training cycle
	for(int i=0;i<1;i++){
		std::cout << "desiredOutput[0]: "  << desiredOutput[i] << std::endl;
		std::cout << "actualOutput[2][i]: "  << layers[2][i] << std::endl;
	}
}

unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){
	//classification code goes here
	cout << "classifying image!\n";
	return 0;
}


vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
