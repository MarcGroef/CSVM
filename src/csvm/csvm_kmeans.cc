#include <csvm/csvm_kmeans.h>
#include <stdlib.h>


using namespace std;
using namespace csvm;


//here we (attempt) to initialize our prototype centroids.
vector<ClusterCentroid> KMeans::initPrototypes(vector<Feature> featureSamples, unsigned int nClusters) {
	
	int nFeatureSamplesSize = featureSamples.size();
	int featureLength = featureSamples[0].size;

	// dirty hack to work with initialization of vector clustercentroids
	ClusterCentroid initial = { Feature(featureLength,0), Feature(featureLength,0), 0 };
	vector<ClusterCentroid> centroids (nClusters, initial); //(featureLength);
	centroids.reserve(nClusters);
	// \hack


	//for every cluster..
	for (size_t idx = 0; idx < nClusters; ++idx)
	{
		//assign the position of a random feature...		
		centroids[idx].newPosition = featureSamples[ rand() % nFeatureSamplesSize];
		//resetcluster copies content from newposition to lastposition, and assigns 0 to nAssignments
		centroids[idx].resetCluster();
	}
	return centroids;
}



vector<Feature> KMeans::cluster(vector<Feature> featureSamples, unsigned int nClusters){
   
	//cout << "we got into clustering!" << '\n';

    unsigned int featureDims = featureSamples[0].size;
    
	//initialize centroids
    vector<ClusterCentroid> centroids = initPrototypes(featureSamples, nClusters);
	//cout << "flag 1" << '\n';
    
	unsigned int featureSampleSize = featureSamples.size();
    bool centroidsChanged = 1;	//track whether any of the centroids have changed
	double smallestDistance = 999999999;	//used to compare the smallest Distances between several clusters and a single feature.

   //cout << "print cluster 0 at start";
   //centroids[0].printValues();

	//while there is a change in the centroids..
   while (centroidsChanged)
   {
	   /*cout << "\nnext iteration\n";
		   for(int idx = 0; idx < centroids.size(); ++idx) {
			   cout << "\ncentroid " << idx << "values : ";
			   centroids[idx].printValues();
		   }
		   //cout << "started clustering" << '\n';
		   */
	   //reset the bool variable
	   centroidsChanged = 0;

	   //iterate through all features.
	   for(size_t fidx = 0; fidx < featureSampleSize; ++fidx)
	   {
		   //for every feature...
			
		   int newWinningCentroid; //used to track the index of the winning centroid
		   size_t clusterLabel;
		   smallestDistance = 999999999;
		   //iterate all clusters (by index clusterLabel)...
		   for (clusterLabel = 0; clusterLabel < centroids.size(); ++clusterLabel)
		   {
			   //and determine label of winning cluster using the distanceSq as indicated
			   double featureDistance = featureSamples[fidx].getDistanceSq( &centroids[clusterLabel].lastPosition );
			   if (featureDistance < smallestDistance)
			   {
				   newWinningCentroid = clusterLabel;
				   smallestDistance = featureDistance;
			   }
		   }
		   //here we should have determined the closest cluster to the feature 'feature'
		   //now we iteratively add the feature vector values to the winning centroid. 
		   centroids[newWinningCentroid].assignFeature(featureSamples[fidx]);
	   }
	   //cout << "first clustering iteration of all features to clusters, now recomputing the new mean for clusters" << '\n';
	   //now we have assigned all features to clusters. So now we recompute the mean for every cluster.
	   
	   //cout << "\ncentroids: ";
	   for (size_t idx = 0; idx < nClusters; ++idx)
	   {
		   //recompute mean
		   centroids[idx].computeNewPosition();

		   //if any of the centroids has changed, then indicate so..
		   
		   if (centroids[idx].hasChanged())
		   {
			   centroidsChanged = 1;
			   //cout << idx << ", ";
		   }
		  
		   //re-initialize the cluster for next iteration.
		   centroids[idx].resetCluster();
	   }
	   //cout << " have changed" << '\n';
   }

   cout << "clustering succeeded";
   //here the means of the centroids have not changed anymore, so we return the feature vectors.

   vector<Feature> finalClusters(nClusters,  (Feature(featureDims,0)));
   finalClusters.reserve(nClusters);
   for (size_t idx = 0;idx < nClusters;++idx)
   {
	   finalClusters[idx] = centroids[idx].lastPosition;
   }
   cout << " returned results " << '\n';
   return finalClusters;
}



