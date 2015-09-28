
#ifndef CSVM_KMEANS_H
#define CSVM_KMEANS_H

namespace csvm{
  
  struct KMeans_settings{
    int nClusters;
  }
  
  class KMeans{
    KMeans_settings settings;
  };
}

#endif
