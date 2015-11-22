#ifndef CSVM_SVM_H
#define CSVM_SVM_H

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
  
  enum SVM_Type{
      CLASSIC,
      CONV,
  };
  
  struct SVM_Settings{
      double SVM_C_Data;
      double SVM_C_Centroid;
      double learningRate;
      double sigmaClassicSimilarity;
      double cost;
      double D2;
      unsigned int nIterations;
      double alphaCentroidInit;
      double alphaDataInit;
      SVM_Kernel kernelType;
      SVM_Type type;
  };
   
  class SVM{
      
    //State variables
    SVM_Settings settings;

    
    unsigned int classId;  
    
    unsigned int nCentroids, nClasses, datasetSize;
    
    
    //internal state
    vector <double> alphaData;
    vector<vector <double> > alphaCentroids;
    double bias;
      
    
    //functions Convolutional SVM    
      double updateAlphaData(vector<vector<double> >& clActivations, unsigned int dataIdx, CSVMDataset* ds);
      double updateAlphaCentroid(vector< vector< vector<double> > >& clActivations, unsigned int centrClass, int centr, CSVMDataset* ds);
      void constrainAlphaCentroid(vector< vector< vector<double> > >& activations);
      void contstrainAlphaData(vector< vector< vector<double> > >& activations, CSVMDataset* ds);
      
    //functions for KKT-SVM
      double constrainAlphaDataClassic(vector< vector<double> >& simKernel, CSVMDataset* ds);
      double updateAlphaDataClassic(vector< vector<double> >& simKernel, CSVMDataset* ds);
      void calculateBiasClassic(vector<vector<double> >& simKernel, CSVMDataset* ds);
      
  public:
     
     SVM(int datasetSize, int nClusters, int nCentroids, unsigned int labelId);
     void setSettings(SVM_Settings s);
     //functions for the convolutional svm
     void train(vector< vector<vector<double> > >& activations, CSVMDataset* ds);
     double classify(vector<vector<double> > f, Codebook* cb);
     
     //functions for the textbook KKT-SVM
     void trainClassic(vector<vector<double> >& simKernel, CSVMDataset* ds);
     double classifyClassic(vector< vector<double> > f, vector< vector< vector<double > > >& datasetActivations, CSVMDataset* cb);
     
     
  };
   
}

#endif