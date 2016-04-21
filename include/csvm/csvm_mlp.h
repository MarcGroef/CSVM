#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <iostream>
#include <vector>
#include "csvm_feature.h"


using namespace std;

namespace csvm{
 
  enum VotingType{
    MAJORITY,
    SUM,
  };
   
   struct MLPSettings{
      //add your settings variables here (stuff you want to set through the settingsfile)
      int nOutputUnits;
      int nHiddenUnits;
      int nInputUnits;
      int nLayers;
      double learningRate;
      string voting;
      string testing;
      int crossValidationInterval;
      double crossValidationSize;
      int epochs;
      double stoppingCriterion;
   };

   class MLPerceptron{
      private:
      //class variables
      MLPSettings settings;
      int noPatchesPerImage;
      
      
   public:
     
      vector<Feature>& normalizeInput(vector<Feature>& allInputFeatures); 	
      void train(vector<Feature>& randomFeatures,vector<Feature>& validationSet, int noPatchPerIm);
      void setSettings(MLPSettings s);
      double fRand(double fMin, double fMax);
      void randomizeWeights(std::vector<vector<double> >& array, int indexBottomLayer);
      double activationFunction(double summedActivation);
      void adjustWeights(int index);
      void calculateError();
      void calculateActivationLayer(int bottomLayer);
      void feedforward();
      void initializeVectors();
      double derivativeActivationFunction(double activationNode);
      double errorFunction();
      void backpropgation();
      void setDesiredOutput(Feature f);
      void hiddenDelta(int index);
      void outputDelta();
      void calculateDeltas(int index);
      
      
      void activationsToOutputProbabilities();
      void voting();
      void majorityVoting();
      void sumVoting();
      unsigned int mostVotedClass();
      
      void training(vector<Feature>& randomFeatures,vector<Feature>& validationSet);
      void crossvaldiation(vector<Feature>& randomFeatures,vector<Feature>& validationSet);
      bool isErrorOnValidationSetLowEnough(vector<Feature>& validationSet);
      void printingWeights();
	  
      unsigned int classify(vector<Feature> imageFeatures);
   };
      
}
#endif
