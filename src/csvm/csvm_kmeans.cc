#include <csvm/csvm_kmeans.h>
//#include <stdlib.h>


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

vector<Feature> KMeans::initCentroids(vector<Feature> collection, unsigned int nClusters){
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   vector<Feature> dictionary(nClusters,Feature(featureSize, 0));
   
   //getchar();
      
   unsigned int randomInt;
   //cout <<"initProto's\n";
   
   for(size_t idx = 0; idx < nClusters; ++idx){
      randomInt = rand() % collectionSize;
      
      while(randomInt < 0) randomInt += collectionSize;
      
      for(size_t d = 0; d < collection[0].content.size(); ++d){
         
         dictionary[idx].content[d] = collection[randomInt].content[d];
      }
   }
   
   return dictionary;

}

void checkEqualFeatures(vector< Feature>& dictionary){
   //cout << "Begin sanity meditation. I see " << dictionary.size() << " features\n";
   unsigned int dictSize = dictionary.size();
   double dist = 0; 
   double delta;
   unsigned int wordSize = dictionary[0].content.size();
   unsigned int nEquals = 0;
   
   
   for(size_t word = 0; word < dictSize ; ++word){
      
      for(size_t word1 = 0; word1 < dictSize; ++word1){
         dist = 0.0f;
         if(word1==word) continue;
         
         for(size_t d = 0; d < wordSize; ++d){
            delta = (dictionary[word].content[d] - dictionary[word1].content[d]);
            //cout << "delta = " << (dictionary[word].content[d] - dictionary[word1].content[d]) << endl;
            dist += delta < 0 ? delta * -1 : delta;
            //cout << "now dist = " << dist << endl;
         }
         //cout << "dist = " << dist << endl;;
         if(dist <= 0) ++nEquals;
      }
      
      
   }
   cout << "I found " << nEquals << " equal features, out of " << dictSize << " features\n";
   
}

vector<Feature> KMeans::cluster(vector<Feature> featureSamples, unsigned int nClusters){
   /*
	//cout << "we got into clustering!" << '\n';

    unsigned int featureDims = featureSamples[0].size;
    
	//initialize centroids
    vector<ClusterCentroid> centroids = initPrototypes(featureSamples, nClusters);
	//cout << "flag 1" << '\n';
    
	unsigned int featureSampleSize = featureSamples.size();
    bool centroidsChanged = 1;	//track whether any of the centroids have changed
	double smallestDistance = 999999999;	//used to compare the smallest Distances between several clusters and a single feature.

   //cout << "print cluster 0 at start";
   centroids[0].printValues();

	//while there is a change in the centroids..
   for (size_t it = 0; centroidsChanged; ++it)
   {
      cout << "kmeans iteration " << it<< "\n";
	  // cout << "\nnext iteration\n";
	//	   for(int idx = 0; idx < centroids.size(); ++idx) {
	//		   cout << "\ncentroid " << idx << "values : ";
	//		   centroids[idx].printValues();
		//   }
		 //  //cout << "started clustering" << '\n';
		   
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
			   double featureDistance = featureSamples[fidx].getDistanceSq(centroids[clusterLabel].lastPosition );
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
   */
   unsigned int nData = featureSamples.size(); 
   vector<Feature> centroids0 = initCentroids(featureSamples, nClusters);
   vector<Feature> centroids1(nClusters, Feature(centroids0[0].content.size(), 0));
   
   vector<Feature>* centroids = &centroids0;
   vector<Feature>* newCentroids = &centroids1;
   int curCentroids = 1;
   unsigned int dataDims = centroids0[0].content.size();
   
   
   vector< unsigned int > nMembers(nClusters,0);
 
   double curDist;
   double prevTotalDistance = 2;
   double totalDistance = 1;
   double deltaDist = 1;
   double closestDist;
   size_t itx = 0;
   for(; /*deltaDist > 0*/ itx < settings.nIter; curCentroids *= -1, ++itx){
      cout << "kmeans iter " << itx << endl;
      prevTotalDistance = totalDistance;
      totalDistance = 0.0;
      
      centroids = (curCentroids == 1 ? &centroids0: &centroids1);
      newCentroids = (curCentroids == 1 ? &centroids1 : &centroids0);
      
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
         for(size_t dim = 0; dim < dataDims; ++dim)
            (*newCentroids)[cIdx].content[dim] = 0.0;
         nMembers[cIdx] = 0;
      }
         
      for(size_t dIdx = 0; dIdx < nData; ++dIdx){
         closestDist = numeric_limits<double>::max();
         unsigned int closestCentr = -1;
         
         for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
            curDist = (*centroids)[cIdx].getDistanceSq(featureSamples[dIdx]);
            //cout << "curDist = " << curDist << endl;
            if(curDist < closestDist){
               closestDist = curDist;
               closestCentr = cIdx;
               
            }
         }
         //cout << "closestDist = " << closestDist << endl;
         //cout << "closestCentr =" << closestCentr << " at dist: " << closestDist << endl;
         totalDistance += closestDist;
         
         for(size_t dim = 0; dim < dataDims; ++dim)
            (*newCentroids)[closestCentr].content[dim] += featureSamples[dIdx].content[dim];
         ++nMembers[closestCentr];
      }
      
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
         if(nMembers[cIdx] > 0)
            for(size_t dim = 0; dim < dataDims; ++dim)
               (*newCentroids)[cIdx].content[dim] /= nMembers[cIdx];
         
      }
      deltaDist = (prevTotalDistance - totalDistance);
      deltaDist = deltaDist < 0 ? deltaDist * -1.0 : deltaDist;
      cout << "deltaDist =  "  << deltaDist << endl;
   }
   
   /*for(size_t centr = 0; centr < nClusters; ++centr){
       cout << "centroid " << centr << " :\n";
       for(size_t idx = 0; idx < dataDims; ++idx){
          cout << (*newCentroids)[centr].content[idx] << ", " << endl;
       }
       cout << endl;
   }*/
   return (*newCentroids);
}




