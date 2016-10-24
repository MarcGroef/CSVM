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
      void Neuron::Neuron(float* inWeights, float* inValues);
      
      float* randomizeWeights();
      float fRand(float fMin, float fMax);
      
      void activationFunction(float summedActivation);
      void sumInputUnits(float* inputValue, float inputWeights*, float b);
      
      float errorFunction(float* desiredOutput, float* actualOutput);
     
   };
      
}


#endif
