#ifndef CSVM_CONV_SVM_H
#define CSVM_CONV_SVM_H

#include <vector>
#include <limits>
#include <iostream>

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
      
   public:
      
      void setSettings(ConvSVMSettings s);
      
      void train(vector< vector<double> >& activations, CSVMDataset* ds);
      unsigned int classify(vector<double>& activations);

   };





}

#endif