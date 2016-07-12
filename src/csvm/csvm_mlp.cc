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
	for( int i = 0; i < layerSizes[indexBottomLayer];i++)
		for(int j = 0; j < layerSizes[indexBottomLayer+1];j++)
			array[i][j] = fRand(-0.5,0.5);
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
	
	momentum = 0.9;
	
	p = 0.5;
	
	if (settings.nLayers == 0){ 
		std::cout << "There is something wrong with the way the settings are set. layerSizes is equal to 0. This should never be the case. " << std::endl;
		exit(-1);
	}
	
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
	prevChange		= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));

	for(int i = 0;i < settings.nLayers-1;i++)
		randomizeWeights(weights[i],i);
}
void MLPerceptron::checkingSettingsValidity(int actualInputSize){
	if(actualInputSize != layerSizes[0]){
		std::cout << "MLP: nInputUnits is set to "<<layerSizes[0]<< ", this is not correct, it should be "<< actualInputSize <<", please change this in the settings file" << std::endl;
		exit(-1);
	}
}

//------end helpFunctions--------
//------start regularization-----
void MLPerceptron::initiateDropOut(int isTraining, int bottomLayer){
    //dropout for trainingphase (input and hidden units are randomly dropped with the propability p)
    //now only hidden units are dropped
    
    if(isTraining && ((bottomLayer+1) < (settings.nLayers-1))){
	double totalAc = 0.0;
	double dropedAc = 0.0;
	
	for(int i=0;i<layerSizes[bottomLayer+1];i++){
	  totalAc += activations[bottomLayer+1][i];  
	}
      
	for(int i=0;i<layerSizes[bottomLayer+1];i++){
	   if(drand48() < p){
	     dropedAc += activations[bottomLayer+1][i];
	     activations[bottomLayer+1][i] = 0.0;
	   }
	}
	//not sure why but it seems like a good idea
	for(int i=0;i<layerSizes[bottomLayer+1];i++){
	    activations[bottomLayer+1][i] = activations[bottomLayer+1][i] * (totalAc/(totalAc-dropedAc));
	}
    }
}

void MLPerceptron::setDropOutTesting(){
  for(unsigned int i =0; i<weights.size();i++){
    for(unsigned int j=0;j<weights[i].size();j++){
      for(unsigned int k=0;k<weights[i][j].size();k++){
	weights[i][j][k] *= p;
      }
    }
  }
}

void MLPerceptron::removeDropOutTesting(){
    for(unsigned int i =0; i<weights.size();i++){
    for(unsigned int j=0;j<weights[i].size();j++){
      for(unsigned int k=0;k<weights[i][j].size();k++){
	weights[i][j][k] /= p;
      }
    }
  }
  
}
//------end regularization-------
//------start FEEDFORWARD--------

double MLPerceptron::activationFunction(double summedActivation){
	//sigmoid:
	//return 1/(1+exp(-summedActivation));
	
	//relu:
	if(summedActivation > 0)
		return summedActivation;
	return 0.0;
  
	//leaky-relu:
	//if(summedActivation > 0)
	//	return summedActivation;
	//return 0.01*summedActivation;
}

void MLPerceptron::calculateActivationLayer(int isTraining, int bottomLayer){
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
	
	initiateDropOut(isTraining,bottomLayer);
}

void MLPerceptron::feedforward(int isTraining){
	for(int i=0;i<settings.nLayers-1;i++)
		calculateActivationLayer(isTraining,i);
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
	
	//leaky-relu
	//if (activationNode > 0)
	//	return 1.0;
	//return 0.01;
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
	//TODO: momentum term
	for(int i = 0; i < layerSizes[index + 1]; i++){
		for(int j = 0; j < layerSizes[index]; j++){
		        double currentChange = settings.learningRate * deltas[index+1][i] * activations[index][j];
			weights[index][j][i] += currentChange + (momentum*prevChange[index][j][i]);
			prevChange[index][j][i] = currentChange;
		}
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
		activations[settings.nLayers -1][i] = exp(activations[settings.nLayers -1][i]);
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
		std::cout << "This voting type is unknown. Change to a known voting type in the settings file" << std::endl; 
		exit(-1);
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
		if (votingHistogram[i] > voteCounter){  //what happens if two classes have the same amount of votes?
			voteCounter = votingHistogram[i];
			mostVotedClass = i;
		}
	}
	return mostVotedClass;
}

//---------end VOTING-----------
//---------start training--------
void MLPerceptron::training(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	if(settings.trainingType == "CROSSVALIDATION")
		crossvaldiation(randomFeatures,validationSet);
	else {
		std::cout << "This training type is unknown. Change to a known training type in the settings file" << std::endl;
		exit(-1);
	}		
}

void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	double averageError = 0;
	int epochs = settings.epochs;
	std::cout << "epochs: " << epochs << std::endl;
	std::cout << "epoch,validationError, averageError" << std::endl;
	
	
	for(int i = 0; i<epochs;i++){
		//cout << " learning rate: " << settings.learningRate << endl;
		
	  	//vector<vector<int> > deadHiddenUnits = vector<vector<int> >(randomFeatures.size(),std::vector<int>(settings.nHiddenUnits,0));
		std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
		
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward(1);
			activationsToOutputProbabilities();
			backpropgation();
			averageError += errorFunction();
			/*
			std::cout << "information weights bottom to first layer" << std::endl;
			for(int k=0;k<weights[0].size();k++){
				for(int l=0;l<weights[0][l].size();l++){
					std::cout << weights[0][k][l] << ",";
				}
				std::cout << std::endl;
			}
			*/
		/*	
			std::cout << "information input units: " << std::endl;
			
			for(int k=0;k<settings.nInputUnits;k++){
				std::cout << activations[0][k] << ", ";	
			}
			std::cout << std::endl;
		
			std::cout << "information hidden units: " << std::endl;

			for(int k=0;k<settings.nHiddenUnits;k++){
				std::cout << activations[1][k] << ", ";	
			}
			std::cout << std::endl;

			std::cout << "information output units: " << std::endl;

			for(int k=0;k<settings.nOutputUnits;k++){
				std::cout << activations[2][k] << ", ";	
			}
			std::cout << std::endl << endl;	
			if(j == 10) exit(-1);
		*//*
		for(int k=0;k<settings.nHiddenUnits;k++)
			if (activations[1][k] != 0) 
				deadHiddenUnits[j][k] = 1;*/
		}
		
		double errorPreviousEpoch = averageError/(double)randomFeatures.size();
		double stopCriterium = 0.0005;
			if(errorPreviousEpoch < stopCriterium){
				cout << "The training process of the mlp stopped because the training error of the previous epoch is " << errorPreviousEpoch << " which is below " << stopCriterium << std::endl;
				cout << "The epoch number is " << i << endl;
				cout << "The validation error at this point is: ";
				isErrorOnValidationSetLowEnough(validationSet);
				cout << endl;
				break;
		}
				
		//after x amount of iterations it should check on the validation set
		if(i % settings.crossValidationInterval == 0 or i == epochs-1){
		  /*int counter = 0;
		  	std::cout << "information input units: " << std::endl;
			
			for(int k=0;k<settings.nInputUnits;k++){
				std::cout << activations[0][k] << ", ";	
			}
			std::cout << std::endl;
		
			std::cout << "information hidden units: " << std::endl;
			
			
			 for(int k=0;k<settings.nHiddenUnits;k++){
				//if (activations[1][k] == 0.0) counter++; //counter amount of zero's
				std::cout << activations[1][k] << ", ";	
			}
			
			std::cout << std::endl;

			std::cout << "information output units: " << std::endl;

			for(int k=0;k<settings.nOutputUnits;k++){
				std::cout << activations[2][k] << ", ";	
			}
			std::cout << std::endl << endl; */
			/*if(i != 0)
				for(int k=0; k<settings.nHiddenUnits;k++){
					int count=0;
					for(int l=i*0.6; l<i;l++){
						if (deadHiddenUnits[l][k] == 0) count++;
					}
					if(count == i-i*0.6) counter++;
				}
			*/
			/*
			for(int k=0; k<settings.nHiddenUnits;k++){
			   int count=0;
			   for(int l=0; l<randomFeatures.size();l++){
			      if (deadHiddenUnits[l][k] == 0) count++;
			    }
			    if(count == randomFeatures.size()) counter++;
			  }*/
			/*
			std::cout << "information bias nodes input to hidden" << std::endl;
			for(int k=0;k<settings.nHiddenUnits;k++){
				std::cout << biasNodes[0][k] << ", ";
			}
			cout << std::endl;
			*/
			//std::cout << "amount of dead hidden units: " << counter << std::endl << endl;
			
			std::cout << i << ", ";
			if(isErrorOnValidationSetLowEnough(validationSet))
				break;
			cout << errorPreviousEpoch << endl;
		}

		settings.learningRate *= 0.98; //decreasing learning rate
	
		averageError = 0;
		
	}
}

void MLPerceptron::training(vector<Feature>& randomFeatures){
	if(settings.trainingType == "CROSSVALIDATION")
		crossvaldiation(randomFeatures);
	else {
		std::cout << "This training type is unknown. Change to a known voting type in the settings file" << std::endl;
		exit(-1);
	}		
}

void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures){
	double averageError = 0;
	int epochs = settings.epochs;
	std::cout << "epochs: " << epochs << std::endl;
	std::cout << "epoch, averageError" << std::endl;
	for(int i = 0; i<epochs;i++){
		//cout << " learning rate: " << settings.learningRate << endl;
		std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
		
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward(1);
			backpropgation();
			averageError += errorFunction();
		}
		cout << i << ", " << averageError << endl;
		averageError = 0;
		
		settings.learningRate *= 0.98;
	}
}

bool MLPerceptron::isErrorOnValidationSetLowEnough(vector<Feature>& validationSet){
	int amountOfImValidationSet = validationSet.size()/numPatchesPerSquare;
	int classifiedCorrect = 0;
	
	//setDropOutTesting();
	
	for(int i = 0; i < amountOfImValidationSet;i++){
		vector<Feature>::const_iterator first = validationSet.begin() + (numPatchesPerSquare *i);
		vector<Feature>::const_iterator last = validationSet.begin() + (numPatchesPerSquare *(i+1));
			
		if(validationSet[i*numPatchesPerSquare].getLabelId() == classify(vector<Feature>(first,last)))
			classifiedCorrect++;
	}
	  
	std::cout << 1.0 - (double)((double)classifiedCorrect/(double)amountOfImValidationSet) << ", ";
	if(classifiedCorrect >= amountOfImValidationSet*settings.stoppingCriterion)
		return 1;
	return 0;
	
	//removeDropOutTesting();
}

void MLPerceptron::train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int numPatchSquare){
	numPatchesPerSquare = numPatchSquare;

	initializeVectors();
	
	checkingSettingsValidity(randomFeatures[0].size);

	cout << "input units: " << settings.nInputUnits << endl;
	cout << "hidden units: " << settings.nHiddenUnits << endl;
	cout << "ouput units: " << settings.nOutputUnits << endl;
	  
	training(randomFeatures,validationSet);			
}

//This method is for training on the validation set
void MLPerceptron::train(vector<Feature>& randomFeatures, int numPatchSquare){
	numPatchesPerSquare = numPatchSquare;
	
	checkingSettingsValidity(randomFeatures[0].size);
		
	training(randomFeatures);	

}
//--------end training-----------
//------start testing------------
void MLPerceptron::classifyImage(vector<Feature>& imageFeatures){
	votingHistogram = vector<double>(settings.nOutputUnits,0.0);
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		activations[0] = imageFeatures[i].content;
		feedforward(0);
		voting();
	}
}
//classify recieves one image in features. It returns the class with the highest activation
unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){
	classifyImage(imageFeatures);		
	return mostVotedClass();
}
vector<double> MLPerceptron::classifyPooling(vector<Feature> imageFeatures){	
	classifyImage(imageFeatures);
	return votingHistogram;
}
void MLPerceptron::returnHiddenActivation(vector<Feature> imageFeatures,vector<double>& maxHiddenActivation){
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		activations[0] = imageFeatures[i].content;
		feedforward(0);
		setMaxActivation(maxHiddenActivation,activations[1]);	
	}
}
void MLPerceptron::setMaxActivation(vector<double>& maxHiddenActivation,vector<double> currentActivation){
	for(unsigned int i=0;i<currentActivation.size();i++){
		if(maxHiddenActivation[i] < currentActivation[i])
			maxHiddenActivation[i] = currentActivation[i];
	}
}

//-------end testing--------
//-------start loading mlp---------
vector<vector<double> > MLPerceptron::getBiasNodes(){
    return biasNodes; 
}

vector<vector<vector<double> > > MLPerceptron::getWeightMatrix(){
    return weights;
}

void MLPerceptron::loadInMLP(vector<vector<vector<double> > > readInWeights,vector<vector<double> > readInBiasNodes){
      initializeVectors();
      
      weights = readInWeights;
      biasNodes = readInBiasNodes;

}
//-------end loading mlp----------
//-------start setters-----------
void MLPerceptron::setWeightMatrix(vector<vector<vector<double> > > newWeights){
    weights = newWeights;
}
//-------end setters -----------
