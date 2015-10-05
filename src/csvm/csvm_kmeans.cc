#include <csvm/csvm_kmeans.h>

using namespace std;
using namespace csvm;



vector<Centroid> KMeans::initPrototypes(vector<Feature> featureSamples, unsigned int nClusters) {
	vector<Centroid> centroids;
	centroids.reserve(nClusters);
	int nFeatureSamplesSize = centroids.size();
	for (size_t idx = 0; idx < nClusters; ++idx)
	{
		centroids[idx].position = featureSamples[rand() % nFeatureSamplesSize];
		centroids[idx].nAssignments = 1;
	}
	return centroids;
}



vector<Feature> KMeans::cluster(vector<Feature> featureSamples, unsigned int nClusters){
   
    unsigned int featureDims = featureSamples[0].size;
    //initialize centroids
    vector<Centroid> centroids = initPrototypes(featureSamples, nClusters);
    unsigned int featureSampleSize = featureSamples.size();
   unsigned int nFeatures = featureSamples.size();

	
   
   
   bool centroidChanged = 0;
   double smallestDistance = 999999999;
   //while there is no change in centroid means...
   while (!centroidChanged)
   {
	   bool centroidChanged = 0;
	   for(size_t fidx = 0; fidx < nFeatures; ++fidx)
	   {
		   int newWinningCentroid; //used to track the index of the winning centroid
		   size_t clusterLabel;
		   //iterate all clusters (by index clusterLabel), and determine label of winning cluster.
		   for (clusterLabel = 0; clusterLabel < centroids.size(); ++clusterLabel)
		   {
			   double featureDistance = featureSamples[fidx].getDistanceSq( &centroids[clusterLabel].position );
			   
			   if (featureDistance < smallestDistance)
			   {
				   newWinningCentroid = clusterLabel;
				   smallestDistance = featureDistance;
			   }
		   }
		   //here we should have determined the closest cluster to the feature 'feature'
		   //now we iteratively add the feature value
		   for (size_t idx = 0;idx < featureDims;++idx) {
			   centroids[newWinningCentroid].position.content[idx] += featureSamples[fidx].content[idx];
		   }
		   centroids[newWinningCentroid].nAssignments += 1;
	   }
   }
		

   vector<Feature> finalClusters;
   finalClusters.reserve(nClusters);
   for (size_t idx = 0;idx < nClusters;++idx)
   {
	   finalClusters[idx] = centroids[idx].position;
   }

   return finalClusters;
}



