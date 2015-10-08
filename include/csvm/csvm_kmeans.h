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
  
  struct ClusterCentroid {
	  Feature lastPosition;
	  Feature newPosition;
	  int nAssignments;
	  void assignFeature(Feature feature)
	  {
		  ++nAssignments;
		  for (int idx = 0; idx < feature.size;++idx)
		  {
			  newPosition.content[idx] += feature.content[idx];
		  }
	  }
	  void computeNewPosition()
	  {
		  for (size_t idx = 0; idx < newPosition.content.size(); ++idx)
		  {
			  newPosition.content[idx] = newPosition.content[idx] / nAssignments;
		  }
	  }
	  
	  void resetCluster()
	  {
		  lastPosition = newPosition;
		  newPosition = new Feature(lastPosition.size, 0);  //Jonathan, free/delete[] je deze alloc wel?
		  nAssignments = 0;
	  }

	  bool hasChanged()
	  {
		  return (newPosition.content != lastPosition.content);
	  }
  };

  class KMeans{
    KMeans_settings settings;
	vector<ClusterCentroid> initPrototypes(vector<Feature> collection, unsigned int nProtos);

    
  public:
    vector<Feature> cluster(vector<Feature> collection, unsigned int nClusters);
  };
}


#endif
