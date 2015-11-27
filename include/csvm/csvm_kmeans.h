#ifndef CSVM_KMEANS_H
#define CSVM_KMEANS_H

#include <limits>
#include <vector>
//#include <iomanip>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
  
  struct KMeans_settings{
    int nClusters;
    double alpha;
    unsigned int nIter;
  };
  
  struct ClusterCentroid {
	  Feature lastPosition;
	  Feature newPosition;
	  int nAssignments;

	  void printValues()
	  {
		  cout << '\n';
		  for (int idx = 0; idx < lastPosition.size; ++idx)
		  {
			  //cout << fixed << setprecision(0) << lastPosition.content[idx]*1000 << " ";
		  }
	  }

	  //assigning a feature directly adds its feature values to the feature of it's to-be new position. 
	  void assignFeature(Feature feature)
	  {
		  ++nAssignments;
		  for (int idx = 0; idx < feature.size;++idx)
		  {
			  newPosition.content[idx] += feature.content[idx];
		  }
	  }

	  //used to compute the new position of the centroid
	  void computeNewPosition()
	  {
		  for (size_t idx = 0; idx < newPosition.content.size(); ++idx)
		  {
			  newPosition.content[idx] = newPosition.content[idx] / nAssignments;
		  }
	  }
	  
	  //updates the position, and resets the newposition. 
	  void resetCluster()
	  {
		  lastPosition = newPosition;
		  newPosition = new Feature(lastPosition.size, 0); //is zonder new niet meer nodig toch? //new Feature(lastPosition.size, 0);  //Jonathan, free/delete[] je deze alloc wel?
		  nAssignments = 0;
	  }

	  bool hasChanged()
	  {
		  for (int idx = 0; idx < lastPosition.size; ++idx) {
			  if (lastPosition.content[idx] != newPosition.content[idx]) {
				  return true;
			  }
		  }
		  return false;
	  }
  };

  class KMeans{
    KMeans_settings settings;
	vector<ClusterCentroid> initPrototypes(vector<Feature> collection, unsigned int nProtos);
   vector<Feature> initCentroids(vector<Feature> collection, unsigned int nClusters);
    
  public:
     void setSettings(KMeans_settings s);
    vector<Feature> cluster(vector<Feature> collection, unsigned int nClusters);
  };
}


#endif
