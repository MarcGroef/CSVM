#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <boost/lexical_cast.hpp>
#include <algorithm>   

using namespace std;
using namespace csvm;


/*
 * All the variables below are declared globally. They are used by all the methods in this class.
 * In the class there is a mix between calling methods with parameters and without.
 * This could lead to confussion.
 */
//-------start variables-------
std::vector<int> layerSizes;

std::vector<double> desiredOutput;
std::vector<double> votingHistogram;

std::vector<vector<double> > biasNodes;
std::vector<vector<double> > activations;
std::vector<vector<double> > deltas;

std::vector<vector<vector<double> > > weights;	
//-------end variables---------

void MLPerceptron::setSettings(MLPSettings s){
   this->settings = s;
   cout << "settings set\n";
}

double good_exp(double y){
	if(y < -10.0) return 0;
	if(y > 10.0) return 9999999.0;
	
	return exp(y);
}

double MLPerceptron::fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void MLPerceptron::randomizeWeights(std::vector<vector<double> >& array,int indexBottomLayer){
	for( int i = 0; i < layerSizes[indexBottomLayer];i++){
		for(int j = 0; j < layerSizes[indexBottomLayer+1];j++)
			array[i][j] = fRand(-0.5,0.5);
	}
}

void MLPerceptron::setDesiredOutput(Feature f){
	int label = f.getLabelId();
	desiredOutput = vector<double>(settings.nOutputUnits,0.0);
	desiredOutput.at(label) = 1;
}
	
double MLPerceptron::errorFunction(){
	double error = 0;
	for (int i=0; i< layerSizes[settings.nLayers-1];i++)
		error += (desiredOutput[i] - activations[settings.nLayers-1][i])*(desiredOutput[i] - activations[settings.nLayers-1][i]);
	error *= 0.5;
	
	return error;
}


//------start FEEDFORWARD--------

double MLPerceptron::activationFunction(double summedActivation){
	//sigmoid:
	//return 1/(1+exp(-summedActivation));
	
	//relu:
	if(summedActivation > 0)
		return summedActivation;
	return 0;
}

void MLPerceptron::calculateActivationLayer(int bottomLayer){
	double summedActivation = 0;
		
	for(int i=0; i<layerSizes[bottomLayer+1];i++){
		for(int j=0;j<layerSizes[bottomLayer];j++)
			summedActivation += activations[bottomLayer][j] * weights[bottomLayer][j][i];
		
		summedActivation += biasNodes[bottomLayer][i];
		if ((bottomLayer+1) == settings.nLayers-1)
			activations[bottomLayer+1][i] = summedActivation;
        else
			activations[bottomLayer+1][i] = activationFunction(summedActivation);
		summedActivation = 0;
	}	
}


void MLPerceptron::feedforward(){
	for(int i=0;i<settings.nLayers-1;i++)
		calculateActivationLayer(i);
}
//--------end FEEDFORWARD--------

//------start BACKPROPAGATION----
double MLPerceptron::derivativeActivationFunction(double activationNode){
	//signmoid:
	//return (1 - activationNode)*activationNode;
	
	//relu
	if (activationNode > 0)
		return 1.0;
	return 0.0;
}

void MLPerceptron::calculateDeltas(int index){
	if (index == 0)
		return;
	if(index == settings.nLayers-1)
		outputDelta();
	if(index > 0 && index < settings.nLayers-1)
		hiddenDelta(index);
	index--;
	return calculateDeltas(index);
}

void MLPerceptron::outputDelta(){
	//ouputDelta without the signmoid, because softmax is used.
	for(int i = 0; i < layerSizes[settings.nLayers-1];i++)
		deltas[settings.nLayers-1][i] = desiredOutput[i] - activations[settings.nLayers-1][i];
}
	
void MLPerceptron::hiddenDelta(int index){
	double sumDeltaWeights = 0;
	for(int i = 0; i < layerSizes[index];i++){
		for(int j = 0; j < layerSizes[index+1];j++)
			sumDeltaWeights += deltas[index+1][j] * weights[index][i][j];
		deltas[index][i] = sumDeltaWeights*derivativeActivationFunction(activations[index][i]);
		sumDeltaWeights = 0;
	}
}	

void MLPerceptron::adjustWeights(int index){
	for(int i = 0; i < layerSizes[index + 1]; i++){
		for(int j = 0; j < layerSizes[index]; j++)
			weights[index][j][i] += settings.learningRate * deltas[index+1][i] * activations[index][j];
		biasNodes[index][i] += settings.learningRate * deltas[index+1][i];
	}
}

void MLPerceptron::backpropgation(){
	calculateDeltas(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-1;i++)
		adjustWeights(i);
}
//--------end BACKPROPAGATION----
//---------start VOTING----------

void MLPerceptron::activationsToOutputProbabilities(){
	double sumOfActivations = 0.0;
	for(int i = 0; i< settings.nOutputUnits; i++){
		activations[settings.nLayers -1][i] = good_exp(activations[settings.nLayers -1][i]);
		sumOfActivations += activations[settings.nLayers -1][i];
	}
	for(int i = 0; i< settings.nOutputUnits; i++)
		activations[settings.nLayers -1][i] /= sumOfActivations;
}


void MLPerceptron::voting(){
	activationsToOutputProbabilities();
	if(settings.voting == "MAJORITY")
		majorityVoting();
	else if (settings.voting == "SUM")
		sumVoting();
	else	{std::cout << "This voting type is unknown. Change to a known voting type in the settings file" << std::endl; exit(-1);}
		
}

void MLPerceptron::majorityVoting(){
	int indexHighestAct = 0;
	double highestActivationClass = 0;
	
	for (int i=0; i<settings.nOutputUnits;i++){
		if(activations[settings.nLayers-1][i]>highestActivationClass){
			highestActivationClass = activations[settings.nLayers-1][i];
			indexHighestAct = i;
		}	
	}
	votingHistogram[indexHighestAct] += 1;
}

void MLPerceptron::sumVoting(){
	for (int i=0; i<settings.nOutputUnits;i++)
			votingHistogram[i] += activations[settings.nLayers-1][i];	
}

unsigned int MLPerceptron::mostVotedClass(){
	unsigned int mostVotedClass = 0;
	double voteCounter = 0;
	
	for (int i = 0; i < settings.nOutputUnits; i++){
		if (votingHistogram[i] > voteCounter){   //what happens if two classes have the same amount of votes?
			voteCounter = votingHistogram[i];
			mostVotedClass = i;
		}
	}
	return mostVotedClass;
}

void MLPerceptron::printingWeights(){
		std::cout << "input-hidden weights: " << std::endl;
		for(int j = 0;j<settings.nInputUnits;j++){
			for(int k = 0;k<settings.nHiddenUnits;k++){
				std::cout << weights[0][j][k] << " ";
				}
			std::cout << std::endl;
			}
		std::cout << std::endl;
			
		std::cout << "hidden-output weights: " << std::endl;
		for(int j = 0;j<settings.nHiddenUnits;j++){
			for(int k = 0;k<settings.nOutputUnits;k++){
				std::cout << weights[1][j][k] << " ";
				}
			std::cout << std::endl;
			}
		std::cout << std::endl;
}

//---------end VOTING-----------

//---------start testing--------
void MLPerceptron::training(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	if(settings.testing == "CROSSVALIDATION"){
		crossvaldiation(randomFeatures,validationSet);
	}else std::cout << "This testing type is unknown. Change to a known voting type in the settings file" << std::endl;		
}

vector<Feature>& MLPerceptron::normalizeInput(vector<Feature>& allInputFeatures){
	double minValue = allInputFeatures[0].content[0];
	double maxValue = allInputFeatures[0].content[0];

	//compute min and max of the all the inputs	
	for(unsigned int i = 0; i < allInputFeatures.size();i++){
		double possibleMaxValue = *std::max_element(allInputFeatures[i].content.begin(), allInputFeatures[i].content.end());
		double possibleMinValue = *std::min_element(allInputFeatures[i].content.begin(), allInputFeatures[i].content.end()); 
		
		if(possibleMaxValue > maxValue)
			maxValue = possibleMaxValue;
			
		if(possibleMinValue < minValue)
			minValue = possibleMinValue;
	}

	if (maxValue - minValue != 0){
		//normalize all the inputs
		for(unsigned int i = 0; i < allInputFeatures.size();i++){
			for(int j = 0; j < allInputFeatures[i].size;j++)
				allInputFeatures[i].content[j] = (allInputFeatures[i].content[j] - minValue)/(maxValue - minValue);
		}
	}else{
		for(unsigned int i = 0; i<allInputFeatures.size();i++){
			for(int j = 0; j < allInputFeatures[i].size;j++)
				allInputFeatures[i].content[j] = 0;
		}
	}
	return allInputFeatures;		
}

void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	double averageError = 0;
	int epochs = settings.epochs;
	std::cout << "epochs: " << epochs << std::endl;
	std::cout << "epoch,validationError, averageError" << std::endl;
	
	for(int i = 0; i<epochs;i++){

		std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
//			std::cout << randomFeatures.at(j).size << std::endl;
			setDesiredOutput(randomFeatures.at(j));
			feedforward();
			backpropgation();
			averageError += errorFunction();
		}
		//after x amount of iterations it should check on the validation set
		if(i % settings.crossValidationInterval == 0){
			std::cout << i << ", ";
			if(isErrorOnValidationSetLowEnough(validationSet))
				break;	
			std::cout << averageError/(double)randomFeatures.size() << std::endl;
		}
		averageError = 0;
	}
}

bool MLPerceptron::isErrorOnValidationSetLowEnough(vector<Feature>& validationSet){
	int amountOfImValidationSet = validationSet.size()/noPatchesPerImage;
	int classifiedCorrect = 0;

	for(int i = 0; i < amountOfImValidationSet;i++){
		vector<Feature>::const_iterator first = validationSet.begin() + (noPatchesPerImage *i);
		vector<Feature>::const_iterator last = validationSet.begin() + (noPatchesPerImage *(i+1));	
		
		if(validationSet[i*noPatchesPerImage].getLabelId() == classify(vector<Feature>(first,last)))
			classifiedCorrect++;
	}
	  
	std::cout << 1.0-(double)((double)classifiedCorrect/(double)amountOfImValidationSet) << ", ";
	if(classifiedCorrect >= amountOfImValidationSet*settings.stoppingCriterion)
		return 1;
		
	return 0;
}


//--------end testing-----------
void MLPerceptron::initializeVectors(){
	int maxNumberOfNodes = 0;
	
	layerSizes		 = vector<int>(settings.nLayers,0);
		
	layerSizes[0] = settings.nInputUnits;
	layerSizes[1] = settings.nHiddenUnits;
	layerSizes[2] = settings.nOutputUnits;
	
	//returns max layer size
	for(unsigned int i = 0; i < layerSizes.size();i++){
		if(maxNumberOfNodes < layerSizes[i])
			maxNumberOfNodes = layerSizes[i];	
	}
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);

	activations		= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	biasNodes 		= vector<vector<double> >(settings.nLayers-1,std::vector<double>(maxNumberOfNodes,0.0));
	
	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));

	for(int i = 0;i < settings.nLayers-1;i++)
		randomizeWeights(weights[i],i);
}

void MLPerceptron::train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int noPatchPerIm){
	noPatchesPerImage = noPatchPerIm;
	
	initializeVectors();

	training(normalizeInput(randomFeatures),normalizeInput(validationSet));			
}

unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){				
	votingHistogram = vector<double>(settings.nOutputUnits,0.0);
	
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		activations[0] = imageFeatures[i].content;
		feedforward();
		voting();
	}
	return mostVotedClass();
}
