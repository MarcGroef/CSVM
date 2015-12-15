#ifndef CSVM_KMEANS_H
#define CSVM_KMEANS_H

#include <limits>
#include <vector>
#include <iostream>
//#include <iomanip>
#include "csvm_feature.h"
#include "csvm_centroid.h"

using namespace std;

namespace csvm{
  
   struct KMeans_settings{
      int nClusters;
      double alpha;
      unsigned int nIter;
   };
  
 

   class KMeans{
      KMeans_settings settings;
      vector<Centroid> initCentroids(vector<Feature> collection, unsigned int nClusters);
      
   public:
      void setSettings(KMeans_settings s);
      vector<Centroid> cluster(vector<Feature>& collection, unsigned int nClusters);
   };
}


#endif
