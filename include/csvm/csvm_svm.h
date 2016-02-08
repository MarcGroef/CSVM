#ifndef CSVM_SVM_H
#define CSVM_SVM_H

/*This class is a class initially built to have a basic working pipeline.
 * The SVM class contains a L2-SVM. Sometimes things are called 'classic', since we 'experimented' a bit with the objective function ed.
 * 
 */

#include <vector>
#include <iostream>
#include "csvm_feature.h"
#include "csvm_codebook.h"
#include "csvm_dataset.h"

using namespace std;
namespace csvm{
  
  enum SVM_Kernel{
      RBF,
      LINEAR,
  };
  

  struct SVM_Settings{
      double SVM_C_Data;
      double learningRate;
      double sigmaClassicSimilarity;
      double cost;
      double D2;
      unsigned int nIterations;
      double alphaDataInit;
      SVM_Kernel kernelType;

  };
   
  class SVM{
      
    //State variables
    SVM_Settings settings;
    unsigned int classId;  
    unsigned int nCentroids, nClasses, datasetSize;

    //internal state
    vector <double> alphaData;
    double bias;
      
    

      
    //functions for KKT-SVM
      double constrainAlphaDataClassic(vector< vector<double> >& simKernel, CSVMDataset* ds);
      double updateAlphaDataClassic(vector< vector<double> >& simKernel, CSVMDataset* ds);
      void calculateBiasClassic(vector<vector<double> >& simKernel, CSVMDataset* ds);
      
  public:
     bool debugOut, normalOut;
     SVM(int datasetSize, int nClusters, int nCentroids, unsigned int labelId);
     void setSettings(SVM_Settings s);
     
     //functions for the textbook KKT-SVM
     void trainClassic(vector<vector<double> >& simKernel, CSVMDataset* ds);
     double classifyClassic(vector<double> f, vector< vector<double > >& datasetActivations, CSVMDataset* cb);
     
     
  };
   
}

#endif