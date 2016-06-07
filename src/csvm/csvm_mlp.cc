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
//-------start helpFunctions------
void MLPerceptron::setSettings(MLPSettings s){
   this->settings = s;
}

double good_exp(double y){
	if(y < -10.0) return 0;
	if(y > 10.0) return 9999999.0;
	
	return exp(y);
}

double fRand(double fMin, double fMax){
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
void MLPerceptron::checkingSettingsValidity(int actualInputSize){
	if(actualInputSize != layerSizes[0]){
		std::cout << "nInputUnits is set to "<<layerSizes[0]<< ", this is not correct, it should be "<< actualInputSize <<", please change this in the settings file" << std::endl;
		exit(-1);
	}
}

void MLPerceptron::setDesiredOutputsForWeighting(vector<double> dofw){
	desiredOutputsForWeighting = dofw;
}

vector<double> MLPerceptron::getDesiredOutputsForWeighting(vector<Feature> featuresForWeightTraining){
	// putting the probability of the right class of a patch in the desired outputs for the weightingMLPs
	if(!desiredOutputsForWeighting.empty())
		desiredOutputsForWeighting.clear();
	for(unsigned int j = 0;j<featuresForWeightTraining.size();j++){
		activations[0] = featuresForWeightTraining.at(j).content;
		feedforward();
		activationsToOutputProbabilities();
		desiredOutputsForWeighting.push_back(activations[settings.nLayers -1][featuresForWeightTraining[j].getLabelId()]);
	}
	return desiredOutputsForWeighting;
}

//------end helpFunctions--------
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
			summedActivation += activations[bottomLayer][j]*weights[bottomLayer][j][i];
		
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

//---------start training--------
void MLPerceptron::training(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	if(settings.trainingType == "CROSSVALIDATION")
		if(settings.isWeightingMLP)
			weightTraining(randomFeatures,validationSet);
		else
			crossvalidation(randomFeatures,validationSet);
	else {
		std::cout << "This training type is unknown. Change to a known training type in the settings file" << std::endl;
		exit(-1);
	}		
}


void MLPerceptron::weightTraining(vector<Feature>& randomFeatures,vector<Feature>& validationSet){   
	double averageError = 0;
	int epochs = settings.epochs;
	std::cout << "epochs: " << epochs << std::endl;
	std::cout << "epoch, \tvalidationError(image), \taverageError" << std::endl;
	
	for(int i = 0; i<epochs;i++){

		//std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
		
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
			desiredOutput[0] = desiredOutputsForWeighting[j];
			feedforward();
			backpropgation();
			averageError += errorFunction();
			/*if( (i+1) == epochs){
				cout << "desiered:\t" << desiredOutput[0] << endl;
				cout << "outputed:\t" << activations[settings.nLayers - 1][0] << endl;
				cout << "\tdifference:\t" << desiredOutput[0] - activations[settings.nLayers - 1][0] << endl;
			}*/
		}
		
		
		
		//after x amount of iterations it should check on the validation set
		if(i % settings.crossValidationInterval == 0){
			std::cout << i << ", \t";
			if(isErrorOnValidationSetLowEnough(validationSet))
				break;	
			std::cout << averageError/(double)randomFeatures.size()*100 << std::endl;
		}
		averageError = 0;
	}
}


void MLPerceptron::crossvalidation(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	double averageError = 0;
	int epochs = settings.epochs;
	std::cout << "epochs: " << epochs << std::endl;
	std::cout << "epoch, \tvalidationError(image), \taverageError" << std::endl;
	
	for(int i = 0; i<epochs;i++){

		//std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
		
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward();
			activationsToOutputProbabilities();
			backpropgation();
			averageError += errorFunction();
		}
		
		//after x amount of iterations it should check on the validation set
		if(i % settings.crossValidationInterval == 0){
			std::cout << i << ", \t";
			if(isErrorOnValidationSetLowEnough(validationSet))
				break;	
			std::cout << averageError/(double)randomFeatures.size() << std::endl;
		}
		averageError = 0;
	}

}

bool MLPerceptron::isErrorOnValidationSetLowEnough(vector<Feature>& validationSet){
	int amountOfImValidationSet = validationSet.size()/numPatchesPerSquare;
	//should be nummerPatchesPerQuarder
	int classifiedCorrect = 0;

	for(int i = 0; i < amountOfImValidationSet;i++){
		vector<Feature>::const_iterator first = validationSet.begin() + (numPatchesPerSquare *i);
		vector<Feature>::const_iterator last = validationSet.begin() + (numPatchesPerSquare *(i+1));	
		
		if(validationSet[i*numPatchesPerSquare].getLabelId() == classify(vector<Feature>(first,last)))
			classifiedCorrect++;
	}
	std::cout << 1.0-(double)((double)classifiedCorrect/(double)amountOfImValidationSet) << ", \t\t\t\t";
	 
	 
	/*double averageValidationError = 0;
	
	for(unsigned int i = 0; i< validationSet.size(); i++){
		activations[0] = validationSet.at(i).content;
		feedforward();
		averageValidationError += errorFunction();
	} 
	std::cout << averageValidationError/(double)validationSet.size() << ", \t\t\t";
	 */
	  
	
	if(classifiedCorrect >= amountOfImValidationSet*settings.stoppingCriterion)
		return 1;
	return 0;
}

void MLPerceptron::train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int numPatchSquare){
	numPatchesPerSquare = numPatchSquare;
	
	initializeVectors();
	
	checkingSettingsValidity(randomFeatures[0].size);
	
	training(randomFeatures,validationSet);			
}

//--------end training-----------
//---------start VOTING----------

//softmax function
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
	else{ 
		std::cout << "This voting type is unknown. Change to a known voting type in the settings file" << std::endl; exit(-1);
	}
		
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

//---------end VOTING-----------
//------start testing------------
//classify recieves one image in features. It returns the class with the highest activation
unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){	
	votingHistogram = vector<double>(settings.nOutputUnits,0.0);
	
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		//std::cout << "Feature label: " << imageFeatures[i].getLabelId() << std::endl;
		activations[0] = imageFeatures[i].content;
		feedforward();
		voting();
	}
	return mostVotedClass();
}

//runFeatureThroughMLP recieves one patch in features. It returns the output of the MLP (e.g. 10 outputs for 10 classes)
vector<double> MLPerceptron::runFeatureThroughMLP(Feature imageFeature){			
	
	activations[0] = imageFeature.content;
	feedforward();
	
	return activations[settings.nLayers - 1];
}
//-------end testing--------
