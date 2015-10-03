#ifndef CSVM_CODEBOOK_H
#define CSVM_CODEBOOK_H

#include "csvm_feature.h"
#include "csvm_lvq.h"
#include "csvm_kmeans.h"

using namespace std;
namespace csvm{
  
  enum CodebookClusterMethod{
    LVQ_Clustering = 0,
    KMeans_Clustering = 1,
    
  };
  
  struct Codebook_settings{
    LVQ_Settings lvqSettings;
    KMeans_settings kmeansSettings;
    CodebookClusterMethod method;
    unsigned int numberVisualWords;
  };
  
  class Codebook{
    Codebook_settings settings;
    LVQ lvq;
    KMeans kmeans;
    vector<Feature> bow;
  public:
    void constructCodeBook(vector<Feature> featureset);
  };
  
}

#endif