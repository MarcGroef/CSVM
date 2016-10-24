#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <iostream>
#include <vector>
#include "csvm_feature.h"


using namespace std;

namespace csvm{
   struct MLPSettings{
      //add your settings variables here (stuff you want to set through the settingsfile)
      int dropout;
      int momentum;
      int nHiddenUnits;
      int nInputUnits;
      int nOutputUnits;
      int nLayers;
      float learningRate;
      string voting;
      string trainingType;
      int crossValidationInterval;
      int epochs;
      float stoppingCriterion;
   };

   class MLPerceptron{
      private:
      //class variables
      MLPSettings settings;
    
      int numPatchesPerSquare;
      int maxNumberOfNodes;
      
      float lapda;
      float p;
      
      vector<int> layerSizes;

	  vector<float> desiredOutput;
	  vector<float> votingHistogram;
	  vector<float> maxHiddenActivation;
	  
	  vector<vector<float> > biasNodes;
          vector<vector<float> > prevBias;
	  vector<vector<bool> > maskBias;
          
          vector<vector<float> > activations;
          vector<vector<float> > prevActiv;
          
	  vector<vector<float> > deltas;
	  vector<vector<float> > prevDeltas;

	  vector<vector<vector<float> > > weights;          	  
          vector<vector<vector<float> > > prevWeights;

	  vector<vector<vector<float> > > prevChange;
          vector<vector<vector<bool> > > mask;
	  
	  //private methods
	  
		//helpMethods:
		void randomizeWeights(vector<vector<float> >& array, int indexBottomLayer);
		void setDesiredOutput(Feature f);
		float errorFunction();
		void initializeVectors();
		void checkingSettingsValidity(int actualInputSize);
		void setHiddenActivationToMethod(vector<float>& hiddenActivation,vector<float>& currentActivation, string type);
		void setDropOutTesting();
		void removeDropOutTesting();
		//regularization:
		void initiateDropOut(int isTraining, int bottomLayer);
		void createMask(int isTraining);
                
		//feedforward:
		float activationFunction(float summedActivation);
		void calculateActivationLayer(int isTraining,int bottomLayer);
		void feedforward(int isTraining);
		//backpropagation:
		float derivativeActivationFunction(float activationNode);
		void calculateDeltas(int index);
		void outputDelta();
		void hiddenDelta(int index);
		void adjustWeights(int index);
		void backpropgation();
		//voting:
		void activationsToOutputProbabilities();
		void voting();
		void majorityVoting();
		void sumVoting();
		unsigned int mostVotedClass();
		//training:
		void training(vector<Feature>& randomFeatures,vector<Feature>& validationSet,vector<Feature>& testSet); 
		void training(vector<Feature>& randomFeatures);
	    
		void crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet,vector<Feature>& testSet);
		void crossvaldiation(vector<Feature>& randomFeatures);
		bool isErrorOnValidationSetLowEnough(vector<Feature>& validationSet);
	  
	  public:
	  void setSettings(MLPSettings s);
	  
          void train(vector<Feature>& randomFeatures,vector<Feature>& validationSet,vector<Feature>& testSet,int numPatchSquare);          	  
          void train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int numPatchSquare);
	  void train(vector<Feature>& validationSet,int numPatchSquare);

          unsigned int classify(vector<Feature> imageFeatures);
	  vector<float> classifyPooling(vector<Feature> imageFeatures);
	  void classifyImage(vector<Feature>& imageFeatures);
	  
	  vector<float> returnHiddenActivationToMethod(vector<Feature> imageFeatures,string type);

      //getters
      vector<float> getMaxActivation();
      vector<vector<float> > getBiasNodes();
      vector<vector<vector<float> > > getWeightMatrix();
      void loadInMLP(vector<vector<vector<float> > > readInWeights, vector<vector<float> > readInBiasNodes);
      
      //setters
      void setWeightMatrix(vector<vector<vector<float> > > newWeights);
      void setEpochs(int epochs);
      void setLearningRate(float learningRate);
   };      
}
#endif