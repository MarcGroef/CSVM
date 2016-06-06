#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <iostream>
#include <vector>
#include "csvm_feature.h"


using namespace std;

namespace csvm{
   
   struct MLPSettings{
      //add your settings variables here (stuff you want to set through the settingsfile)
      int stackSize;
      int nSplitsForPooling;     // if this is 2 -> we have 2x2 pools and therefore 4 MLPs
      int nOutputUnits;
      int nHiddenUnits;
      int nInputUnits;
      int nLayers;
 
      double learningRate;
      string voting;
      string trainingType;
      int crossValidationInterval;
      double crossValidationSize;
      int epochs;
      double stoppingCriterion;
      
      bool useWeightingMLPs;
      bool isWeightingMLP;
      string trainWeightsOn;
      
      int weightingHiddenUnits;
      int weightingLayers;
      
      double weightingLearningRate;
      string weightingVoting;
      int weightingCrossValidationInterval;
      int weightingEpochs;
      double weightingStoppingCriterion;
      
      int saveData;
      string saveRandomFeatName;
      string saveValidationName;
      int readInData;
      string readRandomFeatName;
      string readValidationName;
   };

   class MLPerceptron{
      private:
      //class variables
      MLPSettings settings;
    
      int numPatchesPerSquare;
      

      
      std::vector<int> layerSizes;

			//storage for probabilities of the right class when not isWeightingMLP and source of the desiered outputs if isWeihtingMLP
			std::vector<double> desiredOutputsForWeighting; 
			
			std::vector<double> desiredOutput;
			std::vector<double> votingHistogram;
			std::vector<double> maxHiddenActivation;

			std::vector<vector<double> > biasNodes;
			std::vector<vector<double> > activations;
			std::vector<vector<double> > deltas;

			std::vector<vector<vector<double> > > weights;
	  
	  //private methods
	  
	  //helpMethods:
		void randomizeWeights(std::vector<vector<double> >& array, int indexBottomLayer);
		void setDesiredOutput(Feature f);
		double errorFunction();
		void initializeVectors();
		void checkingSettingsValidity(int actualInputSize);
	  //feedforward:
		double activationFunction(double summedActivation);
		void calculateActivationLayer(int bottomLayer);
		void feedforward();
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
	  void training(vector<Feature>& randomFeatures,vector<Feature>& validationSet);	
	  
	  void weightTraining(vector<Feature>& randomFeatures,vector<Feature>& validationSet); 	
    void crossvalidation(vector<Feature>& randomFeatures,vector<Feature>& validationSet);
		bool isErrorOnValidationSetLowEnough(vector<Feature>& validationSet);

	    
   public:
      void setSettings(MLPSettings s);
      void train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int numPatchSquare);
      vector<double> runFeatureThroughMLP(Feature imageFeature);
      unsigned int classify(vector<Feature> imageFeatures);
      void setDesiredOutputsForWeighting(vector<double> dofw);
      vector<double> getDesiredOutputsForWeighting(vector<Feature> featuresForWeightTraining);
   };
      
}
#endif
