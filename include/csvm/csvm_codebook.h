#ifndef CSVM_CODEBOOK_H
#define CSVM_CODEBOOK_H

#include "csvm_lvq.h"
#include "csvm_kmeans.h"

using namespace std;
namespace csvm{
  
  enum CodebookClusterMethod{
    LVQ_Clustering,
    KMeans_Clustering
    
  };
  
  struct Codebook_settings{
    LVQ_Settings lvqSettings;
    KMeans_settings kmeansSettings;
    CodebookClusterMethod method;
  };
  
  class Codebook{
    Codebook_settings settings;
    LVQ lvq;
    KMeans kmeans;
    
  };
  
}

#endif