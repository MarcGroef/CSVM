#ifndef CSVM_CONV_SVM_H
#define CSVM_CONV_SVM_H

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
      
      int nMax, nMin, answer;
      double maxOut, minOut;
      vector<double> maxOuts;
      vector<double> minOuts;
      vector<double> avOuts;
      vector<double> allOuts;
      ofstream testOutputFile;
      bool writeTestOutput;

   public:
      
      void setSettings(ConvSVMSettings s);
      
      void train(vector< vector< Feature > > dataFeaturesVec, CSVMDataset* ds, Codebook cb);
      unsigned int classify(vector < vector<double> >& activations);
      void setTestOutputFile(string fileName);
      void closeTestOutputFile();
      void setTestAnswer(int a);
   };





}

#endif
