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

double learningRate = 0.4;
	
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
	for(unsigned int i = 0; i < layerSizes[indexLeftLayer]-amountOfBiasNodesLayers[indexLeftLayer];i++){
		for(unsigned int j = 0; j < layerSizes[indexLeftLayer+1]-amountOfBiasNodesLayers[indexLeftLayer+1];j++){
			array[i][j] = fRand(-0.5,0.5);
		}
	}
}

void MLPerceptron::setDesiredOutput(Feature f){
	int label = f.getLabelId();
	desiredOutput = vector<double>(settings.nOutputUnits,0.0);
	desiredOutput.at(label) = 1;
}
	
double MLPerceptron::errorFunction(){
	double error = 0;
	for (int i=0; i< layerSizes[settings.nLayers-1];i++){
		error += (desiredOutput[i] - layers[settings.nLayers-1][i])*(desiredOutput[i] - layers[settings.nLayers-1][i]);
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
	int sizeBiasNodesLeftLayer = amountOfBiasNodesLayers[leftLayerIndex];
	
	for(int i=0; i<rightLayerSize-amountOfBiasNodesLayers[leftLayerIndex+1];i++){
		for(int j=0;j<leftLayerSize-sizeBiasNodesLeftLayer;j++){	
				summedActivation += leftLayer[j] * weights[j][i];
			}

	    for(int j = leftLayerSize-sizeBiasNodesLeftLayer; j < leftLayerSize;j++){
			summedActivation += weights[j][i] * desiredOutput[j-(leftLayerSize-sizeBiasNodesLeftLayer)];
		} 
		rightLayer[i] = activationFunction(summedActivation);
	}	
}

void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++){
		calculateActivationLayer(layerSizes[i],layerSizes[i+1],layers[i],layers[i+1],weights[i],i);
	}
}
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double MLPerceptron::derivativeActivationFunction(double activationNode){
	return (1 - activationNode)*activationNode;
}

void MLPerceptron::calculateError(int index){
	if (index == 0){
		return;
	}
	if(index == settings.nLayers-1){
		outputDelta();
	}
	if(index > 0 && index <= settings.nLayers-2){
		hiddenDelta(index);
	}
	index--;
	return calculateError(index);
}

void MLPerceptron::outputDelta(){
	for(int i = 0; i < layerSizes[settings.nLayers-1];i++){
		deltas[settings.nLayers-1][i] = (desiredOutput[i] - layers[settings.nLayers-1][i])*derivativeActivationFunction(layers[settings.nLayers-1][i]);
	}
}
	
void MLPerceptron::hiddenDelta(int index){
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
	double a_j = 0;
	for(int i = 0; i < sizeRightLayer ; i++){
		for(int j = 0; j < sizeLeftLayer; j++){
			if(j < sizeLeftLayer-amountOfBiasNodesLayers[index]){
				a_j = layers[index][j];  
			}
			else{
				a_j = desiredOutput[ j-(sizeLeftLayer-amountOfBiasNodesLayers[index])];	
			}
			weights[index][j][i] += learningRate * deltas[index+1][i] * a_j;
		}
	}
}

void MLPerceptron::backpropgation(){
	calculateError(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-1;i++){
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
	
	amountOfBiasNodesLayers.at(0) = 10;
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
	
	for(unsigned int i = 0; i < randomFeatures.size();i++){ 
		layers[0] = randomFeatures.at(i).content;
		setDesiredOutput(randomFeatures.at(i));
		
		feedforward();
		backpropgation();
		error = errorFunction();
		std::cout << "error: "  << error << std::endl;
	}
}

void MLPerceptron::test(vector<Feature>& testFeatures){
	
}	


vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
