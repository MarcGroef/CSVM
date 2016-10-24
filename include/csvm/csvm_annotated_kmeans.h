#ifndef CSVM_ANNOTATED_KMEANS_H
#define CSVM_ANNOTATED_KMEANS_H

#include <limits>
#include <vector>
#include <iostream>
#include <iomanip>
#include "csvm_feature.h"
#include "csvm_centroid.h"

//DEPRECATED

using namespace std;

/* Algorithm devised by Jonathan Laurens Maas, @RUG groningen, 25-11-2015
*/

namespace csvm{
	
  struct AKMeans_settings{
    int nClusters;
    float alpha;
	unsigned int nIter;
  };

  enum ContributionFunction {
	  PROPORTIONAL,		//purely based on member distributions
	  STRENGTH,			//also incorporates average distance to mean. Classes closer to a cluster should be represented more

  };

  class AKMeans{
    AKMeans_settings settings;
	vector<vector<float> > clusterByClassContributions;
	//vector<ClusterCentroid> initPrototypes(vector<Feature> collection, unsigned int nProtos);
   vector<Centroid> initCentroids(vector<Feature> collection, unsigned int nClusters, unsigned int nClasses);
   
  public:
     bool debugOut, normalOut;
	  unsigned int nClusters;
	  unsigned int nClasses;
	  void setSettings(AKMeans_settings s);
	  vector<Centroid> clusters;

	  vector< unsigned int > nMembers;	// < number of assigned members	>
	  vector< float > averageDistances;	// < average distance to cluster >
	  vector< float > deviations;	// < standard deviation to center	>

													//per cluster..., per class...
	  vector< vector<unsigned int> > byClassNMembers; //< number of members present in every class >
	  vector< vector< float> > byClassAverageDistancesToCentroid; //average distance to cluster centroid per class
	  vector< vector<float> > byClassDeviationsToCentroid;// deviation to cluster centroid per class


																									  //per classCluster...
	  vector<vector< Centroid > > byClassClusters; // centroids of classes
	  vector<vector< float > > byClassAverageDistancesToClassCluster;	//average distances of class features to classcluster
	  vector<vector<float > > byClassDeviationsToClassCluster;		//deviations of class features to classcluster
	  vector<vector<float> > byClassClusterDistanceToCentroid;



    vector<Centroid> cluster(vector<Feature> collection, unsigned int nClusters, unsigned int nClasses);
	
	vector<vector<float> > getClusterClassContributions();
	vector<float> getClusterClassContributions(int clust);
	vector<float> getClusterClassContributions(Feature feat);

	void printAllClusterStats();
	void printClusterStats(unsigned int clust);


  };
}


#endif
