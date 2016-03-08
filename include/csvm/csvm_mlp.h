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
      void randomizeWeightsInputHidden(std::vector<vector<double> >& array);
      void randomizeWeightsHiddenOutput(std::vector<vector<double> >& array);
      double activationFunction(double summedActivation);
      void calculateActivationLayer(int firstLayerSize,int secondLayerSize,std::vector<double> firstLayer,std::vector<vector<double> > weights,std::vector<double> secondLayer);
      void feedforward();
      double derivativeActivationFunction(double activationNode);
      double errorFunction();
      void adjustWeightsOutputUnits();
      void adjustWeightsHiddenUnit();
      void backpropgation();
      void setDesiredOutput(Feature f);
   };
      
}
#endif
