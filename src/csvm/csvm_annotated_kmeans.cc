#include <csvm/csvm_annotated_kmeans.h>
//#include <stdlib.h>


using namespace std;
using namespace csvm;

/* Algorithm devised by Jonathan Laurens Maas, @RUG groningen, 25-11-2015
*/

//here we (attempt) to initialize our prototype centroids.
/*
vector<ClusterCentroid> AKMeans::initPrototypes(vector<Feature> featureSamples, unsigned int nClusters) {
	
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
*/


void AKMeans::setSettings(AKMeans_settings s) {
	settings = s;
}

vector<Centroid> AKMeans::initCentroids(vector<Feature> featureSamples, unsigned int nClusters, unsigned int nClasses){
   int collectionSize = featureSamples.size();
   unsigned int featureSize = featureSamples[0].size;
   vector<Centroid> initializedClusters(nClusters);
      
   for (size_t clIdx = 0; clIdx < nClusters; ++clIdx) {
	   initializedClusters[clIdx].content.resize(featureSize);
   }

   unsigned int randomInt;
   //cout <<"initProto's\n";
   
   for(size_t idx = 0; idx < nClusters; ++idx){
      randomInt = rand() % collectionSize;
      
      while(randomInt < 0) 
		  randomInt += collectionSize;
      
      for(size_t d = 0; d < featureSize; ++d){
         initializedClusters[idx].content[d] = featureSamples[randomInt].content[d];
      }
   }
   return initializedClusters;
}


/*
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

*/


vector<Centroid> AKMeans::cluster(vector<Feature> featureSamples, unsigned int nClusters, unsigned int nClasses){
	cout << "akmeans called" << endl;
	//unsigned int nClasses = 10;
   unsigned int nData = featureSamples.size(); 
   unsigned int dataDims = featureSamples[0].content.size();

   //centroids0 and centroids1 used to alternate the next array for newCentroids
   vector<Centroid> centroids0 = initCentroids(featureSamples, nClusters, nClasses);
   vector<Centroid> centroids1(nClusters);
   
   

   //byClassNMember used to track, for every cluster, the number of members assigned per class
   vector< vector<int> > byClassNMembers(nClusters, vector<int>(nClasses,0));
   //classavgdistances used to accumulatively gather the average distance to cluster centroid, per class
   vector< vector<double> > byClassAvgDistances(nClusters, vector<double>(nClasses, 0.0));
   //classrepresentativeness is used to represent how much every cluster represents a class.
   vector< vector<double> > byClassContributions(nClusters, vector<double>(nClasses, 0.0));
   
   vector< vector<double> > byClassStrengths(nClusters, vector<double>(nClasses, 0.0));

   //centroids and newCentroids are indicative of current and next position.
   vector<Centroid>* centroids = &centroids0;
   vector<Centroid>* newCentroids = &centroids1;
   
   
   int curCentroids = 1;

   
   
   vector< unsigned int > nMembers(nClusters,0);
 
   double curDist;
   double prevTotalDistance = 2;
   double totalDistance = 1;	//used to track cluster changes in mean
   double deltaDist = 1;
   double closestDist;
   
   for (size_t clIdx = 0; clIdx < nClusters; ++clIdx) {
	   centroids1[clIdx].content.resize(dataDims);
   }
   
   
   
   size_t itx = 0;
   size_t maxIterations = settings.nIter;
   bool lastIteration = 0;



   //7 iterations... I don't like magic numbers though...
   cout << "nIter: " << settings.nIter << endl;
   for(; /*deltaDist > 0*/ itx < settings.nIter; curCentroids *= -1, ++itx){
	   if (itx >= (maxIterations - 1))
		   lastIteration = 1;
      prevTotalDistance = totalDistance;
      
	  totalDistance = 0.0;
      

	  //swap current centroids and new centroids.
      centroids = (curCentroids == 1 ? &centroids0: &centroids1);
      newCentroids = (curCentroids == 1 ? &centroids1 : &centroids0);
      
	  //initialize new centroids 
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
		  for (size_t dim = 0; dim < dataDims; ++dim) {
			  (*newCentroids)[cIdx].content[dim] = 0.0;
		  }
         nMembers[cIdx] = 0;
      }
       
	  //start assigning features to clusters...
      for(size_t dIdx = 0; dIdx < nData; ++dIdx){
         closestDist = numeric_limits<double>::max();
         unsigned int closestCentr = -1;
         
		 //compute closest cluster of a feature
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
         //assign feature to nearest centroid
		 for (size_t dim = 0; dim < dataDims; ++dim) {
			 (*newCentroids)[closestCentr].content[dim] += featureSamples[dIdx].content[dim];
		 }
		 ++nMembers[closestCentr];
		 
		 //only in the last iteration are representational computations relevant
		 if (lastIteration) {
			 byClassAvgDistances[closestCentr][featureSamples[dIdx].labelId] += closestDist;
			 ++byClassNMembers[closestCentr][featureSamples[dIdx].labelId];
		 }
      }
	  

	  //done assigning features to clusters, now we recompute their new centres
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
		  
		  //if a cluster has no members assigned to it
		  if (nMembers[cIdx] <= 0){
			  //leave their centre unchanged
			  //(*newCentroids)[cIdx].content = (*centroids)[cIdx].content;
			  cout << "cluster " << cIdx << " had 0 members" << endl;
		  }
		  else {
			  //compute their location by division of location with number of members
			  for (size_t dim = 0; dim < dataDims; ++dim) {
				  (*newCentroids)[cIdx].content[dim] /= nMembers[cIdx];
			  }
			  if (lastIteration) {
				  double totalClassStrengthSum = 0.0;
				  double currentClassMembers = 0;
				  double currentClassSumOfDistances = 0.0;
				  double currentClassStrength = 0.0;
				  double currentClassAverageDistance = 0.0;
				  double currentClassRepresentativeness = 0.0;
				  for (size_t classID = 0; classID < nClasses; ++classID) {
					  currentClassMembers = byClassNMembers[cIdx][classID];
					  currentClassSumOfDistances = byClassAvgDistances[cIdx][classID];
					  if (currentClassMembers == 0) {
						  //currentClassAverageDistance = 0.0;
						  //currentClassStrength = 0.0;
						  //byClassStrengths[cIdx][classID] = 0.0;
						  byClassContributions[cIdx][classID] = 0.0;
					  }else {
						  currentClassAverageDistance = currentClassSumOfDistances / currentClassMembers;
						  currentClassStrength = currentClassMembers * (1 / currentClassAverageDistance);
						  byClassContributions[cIdx][classID] = currentClassStrength;
						  totalClassStrengthSum += currentClassStrength;
					  }
				  }
				  //now we have computed the total sum of strengths, now normalize to attain proportional representativeness

				  for (size_t classID = 0; classID < nClasses; ++classID) {
					  currentClassMembers = byClassNMembers[cIdx][classID];
					  if (currentClassMembers == 0) {
						  //currentClassAverageDistance = 0.0;
						  //currentClassStrength = 0.0;
						  //byClassStrengths[cIdx][classID] = 0.0;
						  //byClassRepresentativeness[cIdx][classID] = 0.0;
					  }
					  else {
						  //currentClassRepresentativeness /= totalClassStrengthSum;
						  //cout << "cluster " << cIdx << " , class " << classID << " N: " << currentClassMembers << endl;
						  //cout << "  sum of distances: " << currentClassSumOfDistances << endl;
						  //cout << "  current class strength : " << currentClassStrength << endl;
						  //cout << "  sum of strengths of all classes: " << totalClassStrengthSum << endl;
						  //cout << "  representativeness: " << currentClassRepresentativeness << endl;
						  byClassContributions[cIdx][classID] /= totalClassStrengthSum;
					  }
				  }
			  }
		  }
		  
         
      }
      deltaDist = (prevTotalDistance - totalDistance);
      deltaDist = deltaDist < 0 ? deltaDist * -1.0 : deltaDist;
   
   }
   
   for(size_t centr = 0; centr < nClusters; ++centr){
       cout << "centroid " << centr << " represents:\n";
       /*
	   for(size_t idx = 0; idx < dataDims; ++idx){
          cout << (*newCentroids)[centr].content[idx] << ", " << endl;
       }
	   */
	   for (size_t classID = 0; classID < nClasses; ++classID) {
		   cout << "class " << classID << " : " << byClassContributions[centr][classID] << endl;
	   }
       cout << endl;
   }
   clusters = (*newCentroids);
   clusterByClassContributions = byClassContributions;

   return (*newCentroids);
}

vector<vector<double> > AKMeans::getClusterClassContributions() {
	return clusterByClassContributions;
}

vector<double>  AKMeans::getClusterClassContributions(int clust) {
	return clusterByClassContributions[clust];
}



