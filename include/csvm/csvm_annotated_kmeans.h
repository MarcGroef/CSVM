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
    double alpha;
	unsigned int nIter;
  };

  enum ContributionFunction {
	  PROPORTIONAL,		//purely based on member distributions
	  STRENGTH,			//also incorporates average distance to mean. Classes closer to a cluster should be represented more

  };

  class AKMeans{
    AKMeans_settings settings;
	vector<vector<double> > clusterByClassContributions;
	//vector<ClusterCentroid> initPrototypes(vector<Feature> collection, unsigned int nProtos);
   vector<Centroid> initCentroids(vector<Feature> collection, unsigned int nClusters, unsigned int nClasses);
   
  public:
	  unsigned int nClusters;
	  unsigned int nClasses;
	  void setSettings(AKMeans_settings s);
	  vector<Centroid> clusters;

	  vector< unsigned int > nMembers;	// < number of assigned members	>
	  vector< double > averageDistances;	// < average distance to cluster >
	  vector< double > deviations;	// < standard deviation to center	>

													//per cluster..., per class...
	  vector< vector<unsigned int> > byClassNMembers; //< number of members present in every class >
	  vector< vector< double> > byClassAverageDistancesToCentroid; //average distance to cluster centroid per class
	  vector< vector<double> > byClassDeviationsToCentroid;// deviation to cluster centroid per class


																									  //per classCluster...
	  vector<vector< Centroid > > byClassClusters; // centroids of classes
	  vector<vector< double > > byClassAverageDistancesToClassCluster;	//average distances of class features to classcluster
	  vector<vector<double > > byClassDeviationsToClassCluster;		//deviations of class features to classcluster
	  vector<vector<double> > byClassClusterDistanceToCentroid;



    vector<Centroid> cluster(vector<Feature> collection, unsigned int nClusters, unsigned int nClasses);
	
	vector<vector<double> > getClusterClassContributions();
	vector<double> getClusterClassContributions(int clust);
	vector<double> getClusterClassContributions(Feature feat);

	void printAllClusterStats();
	void printClusterStats(unsigned int clust);


  };
}


#endif
