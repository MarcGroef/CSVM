#ifndef CSVM_SVM_H
#define CSVM_SVM_H

#include <vector>
#include <iostream>
#include "csvm_feature.h"
#include "csvm_codebook.h"

using namespace std;
namespace csvm{
  
  
  
  struct SVM_Settings{

  };
   
  class SVM{
      
    //State variables
    SVM_Settings settings;
    vector <double> alphaData;
    vector<vector <double> > alphaCentroids;
    double learningRate;
    unsigned int classId;
    double updateAlphaData(vector<Feature> clActivations, unsigned int dataIdx);
    double updateAlphaCentroid(vector< vector< Feature> > clActivations, unsigned int centrClass, int centr);
    
    vector <double> finalDataWeights;
    unsigned int dataDims;
  public:
     
     SVM(int datasetSize, int nClusters, int nCentroids, double learningRate, unsigned int labelId, int dataDims);
     void train(vector< vector<Feature> > activations);
     int classify(vector<Feature> f, Codebook* cb);
     void contstrainAlphaData(vector< vector< Feature > > activations, unsigned int nIterations, double cost, double maxAlphaVal);
  };
   
}

#endif