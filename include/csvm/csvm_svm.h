#ifndef CSVM_SVM_H
#define CSVM_SVM_H

#include <vector>
#include <iostream>
#include "csvm_feature.h"
#include "csvm_codebook.h"
#include "csvm_dataset.h"

using namespace std;
namespace csvm{
  
  
  
  struct SVM_Settings{
      double SVM_C_Data;
      double SVM_C_Centroid;
      double learningRate;
      double sigmaClassicSimilarity;
  };
   
  class SVM{
      
    //State variables
    SVM_Settings settings;

    
    unsigned int classId;  

    
    //internal state
    vector <double> alphaData;
    vector<vector <double> > alphaCentroids;
    double bias;
      
    
    //functions Convolutional SVM    
      double updateAlphaData(vector<Feature>& clActivations, unsigned int dataIdx);
      double updateAlphaCentroid(vector< vector< Feature> >& clActivations, unsigned int centrClass, int centr);
      void constrainAlphaCentroid(vector< vector< Feature > >& activations, unsigned int nIterations, double cost);
    
    //functions for KKT-SVM
      double constrainAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double cost, unsigned int nIterations);
      double updateAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double D2);
      void calculateBiasClassic(vector<Feature> simKernel, CSVMDataset* ds);
      
  public:
     
     SVM(int datasetSize, int nClusters, int nCentroids, unsigned int labelId);
     void setSettings(SVM_Settings s);
     //functions for the convolutional svm
     void train(vector< vector<Feature> >& activations);
     double classify(vector<Feature> f, Codebook* cb);
     void contstrainAlphaData(vector< vector< Feature > >& activations, unsigned int nIterations, double cost);
     
     //functions for the textbook KKT-SVM
     void trainClassic(vector<Feature> simKernel, CSVMDataset* ds);
     double classifyClassic(vector<Feature> f, vector< vector<Feature> > datasetActivations, CSVMDataset* cb);
  };
   
}

#endif