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

std::vector<vector<double> > finalOutputs; 

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
	for( int i = 0; i < layerSizes[indexLeftLayer];i++){
		for(int j = 0; j < layerSizes[indexLeftLayer+1];j++){
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
	biasNodes 		= vector<vector<double> >(settings.nLayers-1,std::vector<double>(maxNumberOfNodes,0.0));
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);

	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));
	activations		= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	
	for(int i = 0;i < settings.nLayers-1;i++){
		randomizeWeights(weights[i],i);
	}
}

void MLPerceptron::train(vector<Feature>& randomFeatures){
	initializeVectors();

	for(unsigned int i = 0; i < randomFeatures.size();i++){
		activations.at(0) = randomFeatures.at(i).content;
		setDesiredOutput(randomFeatures.at(i));
		feedforward();
		backpropgation();
		error = errorFunction();
	}
}


unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){
	double highestActivationClass;	//activatio of the class with the highest activation
	int outputClassTemp = 0;			//temporary output class, used to find the class a patch is classified to
	int voteCounter = 0;				//counter for which class has the most votes
	unsigned int mostVotedClass = 0;				

	std::vector<int> outputClass(settings.nOutputUnits, 0); //Class histogram for majority voting
	finalOutputs    = vector<vector<double> >(imageFeatures.size(),std::vector<double>(settings.nOutputUnits,0.0));
	
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		activations.at(0) = imageFeatures.at(i).content;
		feedforward();
		finalOutputs.at(i) = activations.at(settings.nLayers-1);
		highestActivationClass = 0;
		for (int j = 0; j<settings.nOutputUnits;j++){
			if(finalOutputs[i][j]>highestActivationClass){
				highestActivationClass = finalOutputs[i][j];
				outputClassTemp = j;
			}	
		}
		outputClass[outputClassTemp] += 1;
	}
	for (int k = 0; k < settings.nOutputUnits; k++){
		if (outputClass[k] > voteCounter){   //what happens if two classes have the same amount of votes?
			voteCounter = outputClass[k];
			mostVotedClass = k;
		}
	}
		
	return mostVotedClass;
}


vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
