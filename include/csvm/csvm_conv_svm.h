#ifndef CSVM_CONV_SVM_H
#define CSVM_CONV_SVM_H

/* Patch-based SVM interpretation. 
 * Concept from M.A. Wiering (2015/2016)
 * 
 * NEEDS CLEANUP FOR RELEASE
 * 
 * 
 */

#include <vector>
#include <limits>
#include <iostream>
#include <sstream>
#include <cmath>

#include "csvm_codebook.h"
#include "csvm_dataset.h"
#include "csvm_image.h"
using namespace std;
namespace csvm{

   struct ConvSVMSettings{
     float learningRate;
     unsigned int nIter;
     float initWeight;
     float CSVM_C;
     unsigned int nClasses;
     unsigned int nCentroids;
     bool L2;
   };
   
   
   class ConvSVM{
      
      ConvSVMSettings settings;
      
      vector< vector<float> > weights;
      vector<float> biases;
    
      float output(vector<float>& activations, unsigned int svmIdx);
      
      int nMax, nMin;
      float maxOut, minOut;
      vector<float> maxOuts;
      vector<float> minOuts;
      vector<float> avOuts;
      vector<float> allOuts;

   public:
      bool debugOut, normalOut;
      void setSettings(ConvSVMSettings s);
      
      void train(vector< vector<float> >& activations, CSVMDataset* ds);
      unsigned int classify(vector<float>& activations);

   };





}

#endif
