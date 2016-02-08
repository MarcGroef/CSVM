#ifndef CSVM_NEW_RBM_H
#define CSVM_NEW_RBM_H

#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include "csvm_feature.h"

using namespace std;

namespace csvm{
   
   struct NRBMSettings{
      unsigned int outputSize;
      unsigned int inputSize;
      unsigned int nGibbsSteps;
      double learningRate;
      unsigned int nIterations;
   };
   
   class NRBM{
      
      NRBMSettings settings;
      vector<double> inputLayer;
      vector<double> outputLayer;
      vector<double> biasUp;
      vector<double> biasDown;
      vector< vector<double> > weights;  //weights[inIdx][outIdx]
      vector< vector<double> > dataEnergy;
      vector< vector<double> > modelEnergy;
      

            
      void flowUp();
      void flowDown();
      double sigmoid(double x);
      void calculateEnergy(vector< vector<double> >& e);
   public:
      bool debugOut, normalOut;
      double studyFeature(vector<double>& f);
      void setSettings(NRBMSettings& s);
      void learn(vector< vector<double> >& data);
      vector<double> describe(vector<double>& f);
      
      
   };
   
   
}

#endif
