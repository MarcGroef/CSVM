#include <csvm/csvm_kmeans.h>

using namespace std;
using namespace csvm;



vector<ClusterCentroid> KMeans::initPrototypes(vector<Feature> featureSamples, unsigned int nClusters) {
	vector<ClusterCentroid> centroids;
	centroids.reserve(nClusters);
	int nFeatureSamplesSize = centroids.size();
	for (size_t idx = 0; idx < nClusters; ++idx)
	{
		centroids[idx].newPosition = featureSamples[rand() % nFeatureSamplesSize];
		centroids[idx].resetCluster();
	}
	return centroids;
}



vector<Feature> KMeans::cluster(vector<Feature> featureSamples, unsigned int nClusters){
   
    unsigned int featureDims = featureSamples[0].size;
    //initialize centroids
    vector<ClusterCentroid> centroids = initPrototypes(featureSamples, nClusters);
    unsigned int featureSampleSize = featureSamples.size();
    unsigned int nFeatures = featureSamples.size();
   bool centroidsChanged = 0;
   double smallestDistance = 999999999;
   //while there is no change in centroid means...
   while (!centroidsChanged)
   {
	   bool centroidChanged = 0;

	   for(size_t fidx = 0; fidx < nFeatures; ++fidx)
	   {
		   int newWinningCentroid; //used to track the index of the winning centroid
		   size_t clusterLabel;
		   //iterate all clusters (by index clusterLabel), and determine label of winning cluster.
		   for (clusterLabel = 0; clusterLabel < centroids.size(); ++clusterLabel)
		   {
			   double featureDistance = featureSamples[fidx].getDistanceSq( &centroids[clusterLabel].lastPosition );
			   
			   if (featureDistance < smallestDistance)
			   {
				   newWinningCentroid = clusterLabel;
				   smallestDistance = featureDistance;
			   }
		   }
		   //here we should have determined the closest cluster to the feature 'feature'
		   //now we iteratively add the feature value
		   centroids[newWinningCentroid].assignFeature(featureSamples[fidx]);

	   }

	   //now we have assigned all features to a cluster. So now we recompute the mean for every cluster.
	   for (size_t idx = 0; idx < nClusters; ++idx)
	   {
		   //recompute mean
		   centroids[idx].computeNewPosition();

		   if (centroids[idx].hasChanged())
		   {
			   centroidChanged = 1;
		   }
		   centroids[idx].resetCluster();
	   }

   }
		

   vector<Feature> finalClusters;
   finalClusters.reserve(nClusters);
   for (size_t idx = 0;idx < nClusters;++idx)
   {
	   finalClusters[idx] = centroids[idx].lastPosition;
   }

   return finalClusters;
}



