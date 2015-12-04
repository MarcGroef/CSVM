#ifndef CSVM_CONV_SVM_H
#define CSVM_CONV_SVM_H

#include <vector>
#include <iostream>

#include "csvm_codebook.h"
#include "csvm_dataset.h"

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
      vector< vector<double> > weights;
      vector<double> biases;
      ConvSVMSettings settings;
      double output(vector< vector<double> >& activations);
   public:
      void setSettings(ConvSVMSettings s);
      void train(vector< vector< vector<double> > >& activations, CSVMDataset* ds);
      
      unsigned int classify(vector< vector<double> >& activations);

   };





}

#endif