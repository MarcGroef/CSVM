#ifndef CSVM_KMEANS_H
#define CSVM_KMEANS_H

#include <vector>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
  
  struct KMeans_settings{
    int nClusters;
    double alpha;
  };
  
  class KMeans{
    KMeans_settings settings;
    
    
  public:
    vector<Feature> cluster(vector<Feature> collection);
  };
}

#endif
