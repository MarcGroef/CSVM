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
double error = 0.0;
std::vector<int> layerSizes;
std::vector<vector<double> >  biasNodes;

std::vector<vector<vector<double> > > weights;
std::vector<vector<double> > activations;

std::vector<vector<double> > deltas;
std::vector<double> desiredOutput;

std::vector<double> testingOuput; 

double learningRate = 0.21;
	
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
	for(unsigned int i = 0; i < layerSizes[indexLeftLayer];i++){
		for(unsigned int j = 0; j < layerSizes[indexLeftLayer+1];j++){
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
		error += (desiredOutput[i] - activations[settings.nLayers-1][i])*(desiredOutput[i] - activations[settings.nLayers-1][i]);
		//std::cout << "desiredOutput[i] - layers[settings.nLayers-1][i]: " << desiredOutput[i] - layers[settings.nLayers-1][i] << std::endl;
	}
	error *= 0.5;
	
	return error;
}


//------start FEEDFORWARD--------

double MLPerceptron::activationFunction(double summedActivation){
	return 1/(1+exp(-summedActivation));
}

void MLPerceptron::calculateActivationLayer(int bottomLayer){
	double summedActivation = 0;
		
	for(int i=0; i<layerSizes[bottomLayer+1];i++){
		for(int j=0;j<layerSizes[bottomLayer];j++){	
				summedActivation += activations[bottomLayer][j] * weights[bottomLayer][j][i];
		}
		summedActivation += biasNodes[bottomLayer][i];
		activations[bottomLayer+1][i] = activationFunction(summedActivation);
		summedActivation = 0;
	}	
}


void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++){
		calculateActivationLayer(i);
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
	if(index > 0 && index < settings.nLayers-1){
		hiddenDelta(index);
	}
	index--;
	return calculateError(index);
}

void MLPerceptron::outputDelta(){
	for(int i = 0; i < layerSizes[settings.nLayers-1];i++){
		deltas[settings.nLayers-1][i] = (desiredOutput[i] - activations[settings.nLayers-1][i])*derivativeActivationFunction(activations[settings.nLayers-1][i]);
	}
}
	
void MLPerceptron::hiddenDelta(int index){
	double sumDeltaWeights = 0;
	//loop over all hidden layer nodes
	for(int i = 0; i < layerSizes[index];i++){
		for(int j = 0; j < layerSizes[index+1];j++){
			sumDeltaWeights += deltas[index+1][j] * weights[index][i][j];
		}
		deltas[index][i] = sumDeltaWeights*derivativeActivationFunction(activations[index][i]);
		sumDeltaWeights = 0;
	}
}	

void MLPerceptron::adjustWeights(int index){
	for(int i = 0; i < layerSizes[index + 1]; i++){
		for(int j = 0; j < layerSizes[index]; j++){
			weights[index][j][i] += learningRate * deltas[index+1][i] * activations[index][j];
		}
		biasNodes[index][i] += learningRate * deltas[index+1][i] * 1;
		//std::cout << "biasNodes[" <<index <<"]["<< i <<"]: " << biasNodes[index][i] <<std::endl;
	}
}


void MLPerceptron::backpropgation(){
	calculateError(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-1;i++){
		adjustWeights(i);
	}
}
//--------end BACKPROPAGATION----


void MLPerceptron::initializeVectors(){
	int maxNumberOfNodes = 0;
	int failure = 0;
	layerSizes		 = vector<int>(settings.nLayers,0);
		
	layerSizes.at(0) = settings.nInputUnits;
	layerSizes.at(1) = settings.nHiddenUnits;
	layerSizes.at(2) = settings.nOutputUnits;
	
	//returns max layer size
	for(unsigned int i = 0; i < layerSizes.size();i++){
		if(maxNumberOfNodes < layerSizes[i]){
			maxNumberOfNodes = layerSizes[i];	
		}
	}
	//Kind of a 'omslachtige' way to make the vectors. There should be a more aligant solution for this
	biasNodes 		 = vector<vector<double> >(settings.nLayers-1,std::vector<double>(maxNumberOfNodes,0.0));
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);

	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));
	activations		= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	testingOuput    = vector<double>(settings.nOutputUnits,0.0);
	
	for(int i = 0;i < settings.nLayers-1;i++){
		randomizeWeights(weights[i],i);
	}
}

void MLPerceptron::train(vector<Feature>& randomFeatures){
	int failure = 0;
	for(int k = 0; k < 10;k++){
	learningRate += 0.01;
	
	for(int j = 0; j<4000;j++){
	//std::cout<< "percentage done: " << (j/1000.0)*100.0 <<"%"<<std::endl;
	initializeVectors();
	//std::cout << "in train, settings.inputLayers: " << settings.nLayers << std::endl;
	
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
	
	for(unsigned int i = 0; i < 60000;i++){ //randomFeatures.size()
		//activations.at(0) = randomFeatures.at(i).content;
		//setDesiredOutput(randomFeatures.at(i));
		
		//testing MLP with XOR
		int num = rand() % 4;
		activations[0] = input[num];
		desiredOutput[0] = possibleOutput[num];
		
		feedforward();
		backpropgation();
		error = errorFunction();

		//errorArray[num] = error;
		/*for(int i=0;i<1;i++){
		std::cout << "desiredOutput[0]: "  << desiredOutput[i] << std::endl;
		std::cout << "actualOutput[2][i]: "  << activations[2][i] << std::endl;
	}*/
		//std::cout << std::endl;
	}
	
		
	for(int i = 0; i < 4;i++){
		int num = i;
		double answer = 0;
		
		activations[0] = input[num];
		desiredOutput[0] = possibleOutput[num];
	
		feedforward();
		answer = round(activations[2][0]);
		
		//std::cout << "activations[2][0];: "<< activations[2][0] << std::endl;
		//std::cout << "desiredOuput: "<< desiredOutput[0] << std::endl;
		
		if(desiredOutput[0] != answer){
			
			//std::cout << "input: "<< activations[0][0] << activations[0][1] << std::endl;
			//std::cout << "failure desiredOutput[0]: "<< desiredOutput[0] << std::endl;
			//std::cout << "answer: "<< answer << std::endl;
			failure++;
			break;
		}
	}
}
std::cout << "failureRate: "  << (failure/4000.0)*100.0 << "%"<< std::endl;
std::cout << "learningRate: "  << learningRate << std::endl;
std::cout << std::endl;
failure = 0;
}
}

unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){
	int failure = 0;
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
	
	for(int i = 0; i < 4;i++){
		int num = i;
		double answer = 0;
		
		activations[0] = input[num];
		desiredOutput[0] = possibleOutput[num];
	
		feedforward();
	
		answer = round(activations[2][0]);
		
		
		if(desiredOutput[0] != answer){
			failure++;
		}
	}
	std::cout << "failureRate: "  << (failure/20000.0)*100.0 << "%"<< std::endl;
	
	//classification code goes here
	cout << "classifying image!\n";
	return 0;
}


vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
