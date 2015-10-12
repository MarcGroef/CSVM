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
    vector<vector <double> > alphaClusters;
    double learningRate;
    
    double updateAlphaData(vector<Feature> data, unsigned int dataIdx, Codebook* cb, unsigned int labelId);
    double kernel(Feature data, Feature centroid);
  public:
     
     SVM(int datasetSize, int nClusters, double learningRate);
     void train(vector<Feature> data, Codebook* cb);
     
      
  };
   
}

#endif