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
  
  struct Centroid {
	  Feature position;
	  Feature lastPosition;
	  Feature newPosition;
	  int nAssignments;
  };

  class KMeans{
    KMeans_settings settings;
	vector<Centroid> initPrototypes(vector<Feature> collection, unsigned int nProtos);

    
  public:
    vector<Feature> cluster(vector<Feature> collection, unsigned int nClusters);
  };
}


#endif
