#include <csvm/csvm_mlp.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
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

void MLPerceptron::randomizeWeights(vector<vector<double> >& array,int indexBottomLayer){
	for( int i = 0; i < layerSizes[indexBottomLayer];i++)
		for(int j = 0; j < layerSizes[indexBottomLayer+1];j++)
			array[i][j] = fRand(-0.1,0.1);
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
	maxNumberOfNodes = 0;
	
	lapda = 0.9;
    
	p = 0.5;
        
	layerSizes             = vector<int>(settings.nLayers,0);
	
	layerSizes[0] = settings.nInputUnits; 
	layerSizes[1] = settings.nHiddenUnits;
	layerSizes[2] = settings.nOutputUnits;
	
        cout << "input units: " << settings.nInputUnits << endl;
        cout << "hidden units: " << settings.nHiddenUnits << endl;
        cout << "ouput unis: " << settings.nOutputUnits << endl;
        
	//returns max layer size
	for(unsigned int i = 0; i < layerSizes.size();i++){
		if(maxNumberOfNodes < layerSizes[i])
			maxNumberOfNodes = layerSizes[i];	
	}
	
	desiredOutput 	= vector<double>(settings.nOutputUnits,0.0);
      
        activations		= vector<vector<double> >(settings.nLayers,vector<double>(maxNumberOfNodes,0.0));
	prevActiv               = vector<vector<double> >(settings.nLayers,vector<double>(maxNumberOfNodes,0.0));
    
        deltas 			= vector<vector<double> >(settings.nLayers,vector<double>(maxNumberOfNodes,0.0));
        prevDeltas              = vector<vector<double> >(settings.nLayers,vector<double>(maxNumberOfNodes,0.0));
        
	biasNodes 		= vector<vector<double> >(settings.nLayers-1,vector<double>(maxNumberOfNodes,0.0));
	maskBias		= vector<vector<bool> >(settings.nLayers-1,vector<bool>(maxNumberOfNodes,0));
	
	weights			= vector< vector< vector<double> > >(settings.nLayers-1,vector< vector<double> >(maxNumberOfNodes, vector<double>(maxNumberOfNodes,0.0)));
        
	for(int i = 0;i < settings.nLayers-1;i++)
		randomizeWeights(weights[i],i);
}
/*
inline int fastrand() { 
  g_seed = (214013*g_seed+2531011); 
  return (g_seed>>16)&0x7FFF; 
} */
#ifndef RAND_SSE_H
#define RAND_SSE_H
#include "emmintrin.h"

//#define COMPATABILITY
//define this if you wish to return values similar to the standard rand();


void srand_sse( unsigned int seed );
void rand_sse( unsigned int* );

static __m128i cur_seed;

void srand_sse( unsigned int seed ) {
    cur_seed = _mm_set_epi32( seed, seed+1, seed, seed+1 );
}

inline void rand_sse( unsigned int* result ) {
     __m128i cur_seed_split;
     __m128i multiplier;
     __m128i adder;
     __m128i mod_mask;
     __m128i sra_mask;
     __m128i sseresult;
     static const unsigned int mult[4] = { 214013, 17405, 214013, 69069 };
     static const unsigned int gadd[4] = { 2531011, 10395331, 13737667, 1 };
     static const unsigned int mask[4] = { 0xFFFFFFFF, 0, 0xFFFFFFFF, 0 };
     static const unsigned int masklo[4] = { 0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF };

    adder = _mm_load_si128( (__m128i*) gadd);
    multiplier = _mm_load_si128( (__m128i*) mult);
    mod_mask = _mm_load_si128( (__m128i*) mask);
    sra_mask = _mm_load_si128( (__m128i*) masklo);
    cur_seed_split = _mm_shuffle_epi32( cur_seed, _MM_SHUFFLE( 2, 3, 0, 1 ) );

    cur_seed = _mm_mul_epu32( cur_seed, multiplier );
    multiplier = _mm_shuffle_epi32( multiplier, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    cur_seed_split = _mm_mul_epu32( cur_seed_split, multiplier );

    cur_seed = _mm_and_si128( cur_seed, mod_mask);
    cur_seed_split = _mm_and_si128( cur_seed_split, mod_mask );
    cur_seed_split = _mm_shuffle_epi32( cur_seed_split, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    cur_seed = _mm_or_si128( cur_seed, cur_seed_split );
    cur_seed = _mm_add_epi32( cur_seed, adder);

    #ifdef COMPATABILITY

    // Add the lines below if you wish to reduce your results to 16-bit vals...
    //sseresult = _mm_srai_epi32( cur_seed, 16);
    //sseresult = _mm_and_si128( sseresult, sra_mask );
    //_mm_storeu_si128( (__m128i*) result, sseresult );
    return;

    #endif

    sseresult = _mm_srai_epi32( cur_seed, 16);
    sseresult = _mm_and_si128( sseresult, sra_mask );
    _mm_storeu_si128( (__m128i*) result, sseresult );
    return;
    //_mm_storeu_si128( (__m128i*) result, cur_seed);
    //return;
}

#endif

void MLPerceptron::checkingSettingsValidity(int actualInputSize){
	if(actualInputSize != layerSizes[0]){
		cout << "MLP:nInputUnits is set to "<<layerSizes[0]<< ", this is not correct, it should be "<< actualInputSize <<", please change this in the settings file" << endl;
		exit(-1);
	}
}

//------end helpFunctions--------
//------start regularization-----
void MLPerceptron::initiateDropOut(int isTraining, int bottomLayer){
    //dropout for trainingphase (input and hidden units are randomly dropped with the propability p)
    //only hidden units are dropped
    
    if(isTraining && ((bottomLayer+1) < (settings.nLayers-1))){
	double activeAc = 0.0;
	double dropedAc = 0.0;
      
	for(int i=0;i<layerSizes[bottomLayer+1];i++){
	   if(drand48() < p){
	     dropedAc += activations[bottomLayer+1][i];
	     activations[bottomLayer+1][i] = 0.0;
	   }
	   else
               activeAc += activations[bottomLayer+1][i];
	}
	for(int i=0;i<layerSizes[bottomLayer+1];i++){
            if(activeAc < 0.01)
                activations[bottomLayer+1][i] = 0.0;
            else
                activations[bottomLayer+1][i] = activations[bottomLayer+1][i] * ((activeAc + dropedAc)/activeAc);
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
        
        //hyperbolic tanget function
        //return (2/(1+exp((-2)*summedActivation)))-1;
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
        if(settings.dropout) 
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
        
        //hyperbolic tanget function
        //return 1 - (activationNode * activationNode);
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
    if(settings.momentum)
        for(int i = 0; i < layerSizes[index + 1]; i++){
            for(int j = 0; j < layerSizes[index]; j++){
                    double currentChange = settings.learningRate * deltas[index+1][i] * activations[index][j];
                    double prevChange = settings.learningRate * prevDeltas[index+1][i] * prevActiv[index][j];
                    weights[index][j][i] += currentChange + lapda * prevChange;
            }
            biasNodes[index][i] += settings.learningRate * deltas[index+1][i];
        }
    else
        for(int i = 0; i < layerSizes[index + 1]; i++){
            for(int j = 0; j < layerSizes[index]; j++){
                    double currentChange = settings.learningRate * deltas[index+1][i] * activations[index][j];
                    weights[index][j][i] += currentChange;
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
	double sumOfActivations = 0;
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
	if(settings.voting == "MAJORITY")
		majorityVoting();
	else if (settings.voting == "SUM")
		sumVoting();
	else{ 
		cout << "This voting type is unknown. Change to a known voting type in the settings file" << endl; 
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
	
	for (int i = 0; i < settings.nOutputUnits; i++)
		if (votingHistogram[i] > voteCounter){  //what happens if two classes have the same amount of votes?
			voteCounter = votingHistogram[i];
			mostVotedClass = i;
		}
	return mostVotedClass;
}

//---------end VOTING-----------
//---------start training--------
void MLPerceptron::crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet, vector<Feature>& testSet){
	double averageError = 0;
	int epochs = settings.epochs;
	cout << "epochs: " << epochs << endl;
        if(!testSet.empty())
            cout << "epoch, validationError, testSetError, averageError" << endl;
	else
            cout << "epoch, validationError, averageError" << endl;

	
	for(int i = 0; i<epochs;i++){
                random_shuffle(randomFeatures.begin(), randomFeatures.end());		
		for(unsigned int j = 0;j<randomFeatures.size();j++){
			activations[0] = randomFeatures.at(j).content;
			setDesiredOutput(randomFeatures.at(j));
			feedforward(1);
			activationsToOutputProbabilities();
			backpropgation();
                        prevDeltas = deltas;
                        prevActiv = activations;
			averageError += errorFunction();
		}
		
		double errorPreviousEpoch = averageError/(double)randomFeatures.size();
				
		//after x amount of iterations it should check on the validation set
		if(i % settings.crossValidationInterval == 0 or i == epochs-1){
			cout << i << ", ";
			if(isErrorOnValidationSetLowEnough(validationSet) || (!testSet.empty() && isErrorOnValidationSetLowEnough(testSet)))
				break;
			cout << errorPreviousEpoch << endl;
		}

		settings.learningRate *= 0.98; //decreasing learning rate
	
		averageError = 0;
		
	}

}

void MLPerceptron::crossvaldiation(vector<Feature>& validationSet){
	double averageError = 0;
	int epochs = settings.epochs;
	
        cout << "epochs: " << epochs << endl;
	cout << "epoch, averageError" << endl;
        cout << " learning rate: " << settings.learningRate << endl;

	for(int i = 0; i<epochs;i++){
		random_shuffle(validationSet.begin(), validationSet.end());
		for(unsigned int j = 0;j<validationSet.size();j++){
			activations[0] = validationSet.at(j).content;
			setDesiredOutput(validationSet.at(j));
			feedforward(1);
                        activationsToOutputProbabilities();
			backpropgation();
			averageError += errorFunction();
                }
		cout << i << ", " << averageError/(double)validationSet.size() << endl;
		averageError = 0;
		
		settings.learningRate *= 0.98;
	}
}

bool MLPerceptron::isErrorOnValidationSetLowEnough(vector<Feature>& validationSet){
	int amountOfImValidationSet = validationSet.size()/numPatchesPerSquare;
	int classifiedCorrect = 0;
	for(int i = 0; i < amountOfImValidationSet;i++){
		vector<Feature>::const_iterator first = validationSet.begin() + (numPatchesPerSquare *i);
		vector<Feature>::const_iterator last = validationSet.begin() + (numPatchesPerSquare *(i+1));
			
		if(validationSet[i*numPatchesPerSquare].getLabelId() == classify(vector<Feature>(first,last)))
			classifiedCorrect++;
	}
        cout << 1.0 - (double)((double)classifiedCorrect/(double)amountOfImValidationSet) << ", ";
	if(classifiedCorrect >= amountOfImValidationSet*settings.stoppingCriterion)
		return 1;
	return 0;
}

//This method is to train randompatches, it uses the validation set and the test set for validation.
void MLPerceptron::train(vector<Feature>& randomFeatures,vector<Feature>& validationSet,vector<Feature>& testSet, int numPatchSquare){
	numPatchesPerSquare = numPatchSquare;
        initializeVectors();
        
        checkingSettingsValidity(randomFeatures[0].size);
        
        crossvaldiation(randomFeatures,validationSet,testSet);
}

//This method is for training on the randomFeatures and testing on the validation set.
//Or this method is for training on the validation set and validating on the test set
//It depends on the parameters being passed.

void MLPerceptron::train(vector<Feature>& trainingSet,vector<Feature>& validationSet,int numPatchSquare){
	numPatchesPerSquare = numPatchSquare;
        vector<Feature> emptyFeatureSet; // empty feature set so the crossvaldiation method does not have to be overloaded again
        
        initializeVectors();
        
        checkingSettingsValidity(trainingSet[0].size);
        
        crossvaldiation(trainingSet,validationSet,emptyFeatureSet); 	
}

//This method is for training on the validation set
void MLPerceptron::train(vector<Feature>& validationSet, int numPatchSquare){
        numPatchesPerSquare = numPatchSquare;
	
        checkingSettingsValidity(validationSet[0].size);
        
        crossvaldiation(validationSet);
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
//The pooling method is set in the settings file. Either Min, average or max.
vector<double> MLPerceptron::returnHiddenActivationToMethod(vector<Feature> imageFeatures, string type){
	int nHiddenUnits = settings.nHiddenUnits;
	vector<double> hiddenActivation;
	
	if(type == "MAX")
		hiddenActivation = vector<double>(nHiddenUnits,-10.0);
				
	else if(type == "AVERAGE")
		hiddenActivation = vector<double>(nHiddenUnits,0.0);
				
	else if(type == "MIN")
		hiddenActivation = vector<double>(nHiddenUnits,50.0);
			
	for (unsigned int i = 0; i<imageFeatures.size();i++){
		activations[0] = imageFeatures[i].content;
		feedforward(0);
		setHiddenActivationToMethod(hiddenActivation,activations[1],type);	
	}
	if(type == "AVERAGE")
		for(unsigned int i=0;i<hiddenActivation.size();i++)
			hiddenActivation[i] /= imageFeatures.size();
			
	return hiddenActivation;
}
void MLPerceptron::setHiddenActivationToMethod(vector<double>& hiddenActivation,vector<double>& currentActivation,string type){
	if(type == "MAX")
		for(unsigned int i=0;i<hiddenActivation.size();i++)
			if(hiddenActivation[i] < currentActivation[i])
				hiddenActivation[i] = currentActivation[i];
	if(type == "AVERAGE")
		for(unsigned int i=0;i<hiddenActivation.size();i++)
			hiddenActivation[i] += currentActivation[i];		
	if(type == "MIN")
		for(unsigned int i=0;i<hiddenActivation.size();i++)
			if(hiddenActivation[i] > currentActivation[i])
				hiddenActivation[i] = currentActivation[i];
}

//-------end testing---------------
//-------start loading mlp---------
vector<vector<double> > MLPerceptron::getBiasNodes(){
    return biasNodes; 
}

vector<vector<vector<double> > > MLPerceptron::getWeightMatrix(){
    return weights;
}

void MLPerceptron::loadInMLP(vector<vector<vector<double> > > readInWeights,vector<vector<double> > readInBiasNodes)
{
      initializeVectors();
      
      weights = readInWeights;
      biasNodes = readInBiasNodes;

}
//-------end loading mlp----------
//-------start setters-----------
void MLPerceptron::setWeightMatrix(vector<vector<vector<double> > > newWeights)
{
    weights = newWeights;
}

void MLPerceptron::setEpochs(int epochs)
{    
    settings.epochs = epochs; //This is tricky when an mlp is saved.
}

void MLPerceptron::setLearningRate(double learningRate){
    settings.learningRate = learningRate;
}
//-------end setters -----------