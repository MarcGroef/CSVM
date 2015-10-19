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
      double SVM_C;
  };
   
  class SVM{
      
    //State variables
    SVM_Settings settings;
    vector <double> alphaData;
    vector<vector <double> > alphaCentroids;
    double learningRate;
    unsigned int classId;
    double updateAlphaData(vector<Feature>& clActivations, unsigned int dataIdx);
    double updateAlphaCentroid(vector< vector< Feature> >& clActivations, unsigned int centrClass, int centr);
    void constrainAlphaCentroid(vector< vector< Feature > >& activations, unsigned int nIterations, double cost);
    vector <double> finalDataWeights;
    unsigned int dataDims;
  public:
     
     SVM(int datasetSize, int nClusters, int nCentroids, double learningRate, unsigned int labelId, int dataDims);
     void train(vector< vector<Feature> >& activations);
     double classify(vector<Feature> f, Codebook* cb);
     void contstrainAlphaData(vector< vector< Feature > >& activations, unsigned int nIterations, double cost);
     double constrainAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double cost, unsigned int nIterations);
     double updateAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double D2);
     void trainClassic(vector<Feature> simKernel, CSVMDataset* ds);
     double classifyClassic(vector<Feature> f, vector< vector<Feature> > datasetActivations, CSVMDataset* cb);
  };
   
}

#endif