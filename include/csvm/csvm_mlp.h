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
      double learningRate;
      string voting;
      string trainingType;
      int crossValidationInterval;
      int epochs;
      double stoppingCriterion;
   };

   class MLPerceptron{
      private:
      //class variables
      MLPSettings settings;
    
      int numPatchesPerSquare;
      int maxNumberOfNodes;
      
      double lapda;
      double p;
      
      std::vector<int> layerSizes;

	  std::vector<double> desiredOutput;
	  std::vector<double> votingHistogram;
	  std::vector<double> maxHiddenActivation;
	  
	  std::vector<vector<double> > biasNodes;
          std::vector<vector<double> > prevBias;
	  std::vector<vector<bool> > maskBias;
          
          std::vector<vector<double> > activations;
	  std::vector<vector<double> > deltas;

	  std::vector<vector<vector<double> > > weights;          	  
          std::vector<vector<vector<double> > > prevWeights;

	  std::vector<vector<vector<double> > > prevChange;
          std::vector<vector<vector<bool> > > mask;
	  
	  //private methods
	  
		//helpMethods:
		void randomizeWeights(std::vector<vector<double> >& array, int indexBottomLayer);
		void setDesiredOutput(Feature f);
		double errorFunction();
		void initializeVectors();
		void checkingSettingsValidity(int actualInputSize);
		void setHiddenActivationToMethod(vector<double>& hiddenActivation,vector<double>& currentActivation, string type);
		void setDropOutTesting();
		void removeDropOutTesting();
		//regularization:
		void initiateDropOut(int isTraining, int bottomLayer);
		void createMask(int isTraining);
                
		//feedforward:
		double activationFunction(double summedActivation);
		void calculateActivationLayer(int isTraining,int bottomLayer);
		void feedforward(int isTraining);
		//backpropagation:
		double derivativeActivationFunction(double activationNode);
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
	  vector<double> classifyPooling(vector<Feature> imageFeatures);
	  void classifyImage(vector<Feature>& imageFeatures);
	  
	  vector<double> returnHiddenActivationToMethod(vector<Feature> imageFeatures,string type);

      //getters
      vector<double> getMaxActivation();
      vector<vector<double> > getBiasNodes();
      vector<vector<vector<double> > > getWeightMatrix();
      void loadInMLP(vector<vector<vector<double> > > readInWeights, vector<vector<double> > readInBiasNodes);
      
      //setters
      void setWeightMatrix(vector<vector<vector<double> > > newWeights);
      void setEpochs(int epochs);
      void setLearningRate(double learningRate);
   };      
}
#endif