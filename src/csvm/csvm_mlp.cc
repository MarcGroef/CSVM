#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <algorithm>   

using namespace std;
using namespace csvm;


/*
 * All the variables below are declared globally. They are used by all the methods in this class.
 * In the class there is a mix between calling methods with parameters and without.
 * This could lead to confussion.
 */
//-------start variables-------
int sizeRandomFeat;


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

double MLPerceptron::fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void MLPerceptron::randomizeWeights(std::vector<vector<double> >& array,int indexBottomLayer){
	for( int i = 0; i < layerSizes[indexBottomLayer];i++){
		for(int j = 0; j < layerSizes[indexBottomLayer+1];j++){
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

void MLPerceptron::calculateDeltas(int index){
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
	return calculateDeltas(index);
}

void MLPerceptron::outputDelta(){
	for(int i = 0; i < layerSizes[settings.nLayers-1];i++){
		deltas[settings.nLayers-1][i] = (desiredOutput[i] - activations[settings.nLayers-1][i])*derivativeActivationFunction(activations[settings.nLayers-1][i]);
	}
}
	
void MLPerceptron::hiddenDelta(int index){
	double sumDeltaWeights = 0;
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
			weights[index][j][i] += settings.learningRate * deltas[index+1][i] * activations[index][j];
		}
		biasNodes[index][i] += settings.learningRate * deltas[index+1][i];
	}
}

void MLPerceptron::backpropgation(){
	calculateDeltas(settings.nLayers-1);
	for(int i = 0; i < settings.nLayers-1;i++){
		adjustWeights(i);
	}
}
//--------end BACKPROPAGATION----
//---------start VOTING----------

void MLPerceptron::activationsToOutputProbabilities(){
	double sumOfActivations = 0.0;
	for(int i = 0; i< settings.nOutputUnits; i++){
		activations[settings.nLayers -1][i] = exp(activations[settings.nLayers -1][i]);
		sumOfActivations += activations[settings.nLayers -1][i];
	}
	for(int i = 0; i< settings.nOutputUnits; i++){
		activations[settings.nLayers -1][i] /= sumOfActivations;
	}
}


void MLPerceptron::voting(){
	activationsToOutputProbabilities();
	if(settings.voting == "MAJORITY"){
		majorityVoting();
	}else if (settings.voting == "SUM"){
		sumVoting();
		}else{
			std::cout << "This voting type is unknown. Change to a known voting type in the settings file" << std::endl;
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
	for (int i=0; i<settings.nOutputUnits;i++){
			votingHistogram[i] += activations[settings.nLayers-1][i];	
	}
	
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
	}else if (settings.testing == "RERUN"){
		rerun(randomFeatures);
		}else{
			std::cout << "This testing type is unknown. Change to a known voting type in the settings file" << std::endl;
		}
	//printingWeights();
}
void valueToScoreFile(string value){
	std::ofstream myfile;
  myfile.open("train-validation.csv", std::ios_base::app);
  myfile << value << ",";
  myfile.close();
}

void lastInputLine(string value){
	std::ofstream myfile;
	
  myfile.open("train-validation.csv", std::ios_base::app);
  myfile << value << std::endl;
  myfile.close();
}

void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	double averageError = 0;
	int epochs = 100;
	std::cout << "validationError, averageError" << std::endl;
	for(int i = 0; i<epochs;i++){
		std::random_shuffle(randomFeatures.begin(), randomFeatures.end());
	
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations.at(0) = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward();
			backpropgation();
			averageError += errorFunction();
		}
		valueToScoreFile(boost::lexical_cast<string>(averageError/(double)randomFeatures.size()));
		
		if(errorOnValidationSet(validationSet))
			break;	
		
		std::cout << averageError/(double)randomFeatures.size() << std::endl;
		
		averageError = 0;
	}
}

bool MLPerceptron::errorOnValidationSet(vector<Feature>& validationSet){
	int amountOfImValidationSet = validationSet.size()/noPatchesPerImage;
	int classifiedCorrect = 0;
	int patchesPerIm = validationSet.size() / amountOfImValidationSet;


	for(int i = 0; i < amountOfImValidationSet;i++){
		vector<Feature>::const_iterator first = validationSet.begin() + (patchesPerIm *i);
		vector<Feature>::const_iterator last = validationSet.begin() + (patchesPerIm *(i+1));
		
		if(validationSet[i*patchesPerIm].getLabelId() == classify(vector<Feature>(first,last)))
			classifiedCorrect++;
	}
	  
	lastInputLine(boost::lexical_cast<string>(1.0-(double)((double)classifiedCorrect/(double)amountOfImValidationSet)));
	std::cout << 1.0-(double)((double)classifiedCorrect/(double)amountOfImValidationSet) << ", ";
	//std::cout << "correctly classified: " << (double)classifiedCorrect/(double)(validationSet.size()/noPatchesPerImage) * 100 << "%" << std::endl;
	
	if(classifiedCorrect >= amountOfImValidationSet*0.5) //magic number
		return 1;
		
	return 0;
}

/*
void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet){
	int threshold = 0;
	int iterations = 1; 
	double errorValidation = 0;
	double tempErrorValidation = 0;
	vector<int> counterValidation(settings.nOutputUnits,0);
	std::vector<std::vector<double> > errorClasses(settings.nOutputUnits, vector<double>(100,1.0));
	double maxError = 1;
	
	while (threshold != 1){
		std::cout << "number of iterations " << iterations << std::endl;
		iterations++;
		for(unsigned int i = 0; i < randomFeatures.size();i++){
			activations.at(0) = randomFeatures.at(i).content;
			setDesiredOutput(randomFeatures.at(i));
			feedforward();
			if(i<(0.8*randomFeatures.size())){
				backpropgation();
			}else{
				errorValidation	= errorFunction();
				errorClasses[randomFeatures.at(i).getLabelId()][counterValidation[randomFeatures.at(i).getLabelId()] % 100] = errorValidation;
				counterValidation[randomFeatures.at(i).getLabelId()] ++;
				for(int j = 0; j < settings.nOutputUnits;j++){
					if (counterValidation[j] >= 100){	// only calculate average error if there have been 100 patches with that lable in the validation set till now
						tempErrorValidation = 0;
						for(unsigned int k = 0; k < 100; k++){		//average last 100 errors of one class
							tempErrorValidation += errorClasses[j][k];
						}
						tempErrorValidation /= 100;
						if(tempErrorValidation > maxError){
							maxError = tempErrorValidation;
						}
						if(maxError < 0.01){
							std::cout << "max error Validation set" << maxError << std::endl;
							threshold = 1;
						}
					}
				}
				maxError = 0;
			}
		}
	}
}
*/
void MLPerceptron::rerun(vector<Feature>& randomFeatures){
	int epochs = 3;
	//double error = 0;
	votingHistogram = vector<double>(settings.nOutputUnits,1);
	for (int i = 0;i<epochs;i++){
		std::cout << "i: " << i << std::endl;
		for(unsigned int j = 0; j < randomFeatures.size();j++){
			//std::cout << "j: " << j << std::endl;
			activations.at(0) = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward();
			backpropgation();
			
			votingHistogram[randomFeatures.at(j).getLabelId()] = errorFunction();
			
			//std::cout << errorFunction() << "label: " << randomFeatures.at(j).getLabelId() << std::endl;
		}
	}
	for(int i = 0; i < settings.nOutputUnits;i++){
			std::cout << "votingHistogram[" << i <<"]: " << votingHistogram[i] << std::endl;
		} 
		std::cout << std::endl;
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
		if(maxNumberOfNodes < layerSizes[i]){
			maxNumberOfNodes = layerSizes[i];	
		}
	}
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);

	activations		= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	deltas 			= vector<vector<double> >(settings.nLayers,std::vector<double>(maxNumberOfNodes,0.0));
	biasNodes 		= vector<vector<double> >(settings.nLayers-1,std::vector<double>(maxNumberOfNodes,0.0));
	
	weights			= vector< vector< vector<double> > >(settings.nLayers-1,std::vector< vector<double> >(maxNumberOfNodes, std::vector<double>(maxNumberOfNodes,0.0)));

	for(int i = 0;i < settings.nLayers-1;i++){
		randomizeWeights(weights[i],i);
	}
}

void MLPerceptron::train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int noPatchPerIm){
	noPatchesPerImage = noPatchPerIm;
	
	sizeRandomFeat = randomFeatures.size();
	
	valueToScoreFile(boost::lexical_cast<string>(sizeRandomFeat));
	valueToScoreFile(boost::lexical_cast<string>(settings.nHiddenUnits));
	valueToScoreFile(boost::lexical_cast<string>(settings.voting));
	lastInputLine(boost::lexical_cast<string>(settings.learningRate));
	
	initializeVectors();
	training(randomFeatures,validationSet);
			
}

unsigned int MLPerceptron::classify(vector<Feature> imageFeatures){				
	votingHistogram = vector<double>(settings.nOutputUnits,0.0);
	
	for (unsigned int i = 0; i<imageFeatures.size();i++){
//		std::cout << "label[" << i << "]: " << imageFeatures[i].getLabelId() << std::endl;
		activations[0] = imageFeatures[i].content;
		feedforward();
		voting();
	}
	return mostVotedClass();
}


vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
