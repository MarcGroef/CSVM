#ifndef CSVM_NEURON_H
#define CSVM_NEURON_H

#include <iostream>
#include <vector>
#include "csvm_feature.h"
#include "math.h"

using namespace std;

namespace csvm{
   
   struct Neuron{
      //add your settings variables here (stuff you want to set through the settingsfile)
   };

      
      
      
   class Neuron{
   private:
      //class variables
      Neuron settings;
      
   public:
      //void train(vector<Feature>& randomFeatures);
      void Neuron::Neuron(double* inWeights, double* inValues);
      
      double* randomizeWeights();
      double fRand(double fMin, double fMax);
      
      void activationFunction(double summedActivation);
      void sumInputUnits(double* inputValue, double inputWeights*, double b);
      
      double errorFunction(double* desiredOutput, double* actualOutput);
     
   };
      
}


#endif
