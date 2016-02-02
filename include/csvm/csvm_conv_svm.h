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
     double learningRate;
     unsigned int nIter;
     double initWeight;
     double CSVM_C;
     unsigned int nClasses;
     unsigned int nCentroids;
   };
   
   
   class ConvSVM{
      
      ConvSVMSettings settings;
      
      vector< vector<double> > weights;
      vector<double> biases;
    
      double output(vector<double>& activations, unsigned int svmIdx);
      
      int nMax, nMin;
      double maxOut, minOut;
      vector<double> maxOuts;
      vector<double> minOuts;
      vector<double> avOuts;
      vector<double> allOuts;

   public:
      
      void setSettings(ConvSVMSettings s);
      
      void train(vector< vector<double> >& activations, CSVMDataset* ds);
      unsigned int classify(vector<double>& activations);

   };





}

#endif
