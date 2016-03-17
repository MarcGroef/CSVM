#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <iostream>
#include <vector>
#include "csvm_feature.h"


using namespace std;

namespace csvm{
   
   
   struct MLPSettings{
      //add your settings variables here (stuff you want to set through the settingsfile)
      int nOutputUnits;
      int nHiddenUnits;
      int nInputUnits;
      int nLayers;
   };

   class MLPerceptron{
   private:
      //class variables
      MLPSettings settings;
      
      
   public:
     
      void train(vector<Feature>& randomFeatures);
      vector<double> getActivations(vector<Feature>& imageFeatures);
      void setSettings(MLPSettings s);
      double fRand(double fMin, double fMax);
      void randomizeWeights(std::vector<vector<double> >& array, int indexLeftLayer);
      double activationFunction(double summedActivation);
      void adjustWeights(int index);
      void calculateActivationLayer(int leftLayerSize ,int rightLayerSize, std::vector<double> &leftLayer,std::vector<double> &rightLayer, std::vector<vector<double> > weights,int leftLayerIndex);
      void calculateError();
      void feedforward();
      void initializeVectors();
      double derivativeActivationFunction(double activationNode);
      double errorFunction();
      void adjustWeightsOutputUnits();
      void adjustWeightsHiddenUnit();
      void backpropgation();
      void setDesiredOutput(Feature f);
      void adjustWeights(int index, int sizeLeftLayer, int sizeRightLayer);
      void hiddenDelta(int index);
      void outputDelta();
      void calculateError(int index);
	  unsigned int classify(vector<Feature> imageFeatures);

   };
      
}
#endif
