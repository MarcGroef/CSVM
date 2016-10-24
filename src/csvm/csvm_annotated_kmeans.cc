#include <csvm/csvm_annotated_kmeans.h>

//DEPRECATED
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
   float dist = 0; 
   float delta;
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


vector<Centroid> AKMeans::cluster(vector<Feature> featureSamples, unsigned int pnClusters, unsigned int pnClasses){
	cout << "akmeans called" << endl;
	nClusters = pnClusters;
	nClasses = pnClasses;
   unsigned int nData = featureSamples.size();		//number of features to cluster
   unsigned int dataDims = featureSamples[0].content.size();	//feature dimensionality

   //centroids0 and centroids1 used to alternate the next array for newCentroids
   

   //per cluster.. <   Centroid: location	>
   vector<Centroid> centroids0 = initCentroids(featureSamples, nClusters, nClasses);
   vector<Centroid> centroids1(nClusters);
   
   //per cluster.. 
   nMembers = vector< unsigned int >(nClusters, 0);	// < number of assigned members	>
   averageDistances = vector< float >(nClusters, 0.0);	// < average distance to cluster >
   deviations = vector< float >(nClusters, 0.0);	// < standard deviation to center	>

   //per cluster..., per class...
   byClassNMembers = vector< vector<unsigned int> >(nClusters, vector<unsigned int>(nClasses, 0)); //< number of members present in every class >
   byClassAverageDistancesToCentroid = vector< vector< float> >(nClusters, vector<float>(nClasses, 0.0)); //average distance to cluster centroid per class
   byClassDeviationsToCentroid = vector< vector<float> >(nClusters, vector<float>(nClasses, 0.0)); // deviation to cluster centroid per class
									

   //per classCluster...
   //vector<vector< Centroid > > 
   byClassClusters = vector<vector< Centroid> >(nClusters, vector<Centroid>(nClasses)); // centroids of classes
   byClassAverageDistancesToClassCluster = vector<vector< float > >(nClusters, vector<float>(nClasses, 0.0));	//average distances of class features to classcluster
   byClassDeviationsToClassCluster = vector<vector<float > >(nClusters, vector<float>(nClasses, 0.0));		//deviations of class features to classcluster
																												//deviations of class features to classcluster
   byClassClusterDistanceToCentroid = vector<vector<float> >(nClusters, vector<float>(nClasses, 0.0));

   //classrepresentativeness is used to represent how much every cluster represents a class.
   vector< vector<float> > byClassContributions(nClusters, vector<float>(nClasses, 0.0));
   vector< vector<float> > byClassStrengths(nClusters, vector<float>(nClasses, 0.0));



   //centroids and newCentroids are indicative of current and next position.
   vector<Centroid>* centroids = &centroids0;
   vector<Centroid>* newCentroids = &centroids1;
   
   
   int curCentroids = 1;

   
   
 
   float curDist;
   float prevTotalDistance = 2;
   float totalDistance = 1;	//used to track cluster changes in mean
   float deltaDist = 1;
   float closestDist;
   
   for (size_t clIdx = 0; clIdx < nClusters; ++clIdx) {
	   centroids1[clIdx].content.resize(dataDims);
	   for (size_t classId = 0; classId < nClasses; ++classId) {
		   byClassClusters[clIdx][classId].content.resize(dataDims);
	   }

   }
   
   
   
   size_t itx = 0;
   size_t maxIterations = settings.nIter;
   //bool lastIteration = 0;



   //7 iterations... I don't like magic numbers though...
   //cout << "nIter: " << settings.nIter << endl;
   for(; /*deltaDist > 0*/ itx < settings.nIter; curCentroids *= -1, ++itx){
	   cout << "iteration: " << itx << endl;
	   if (itx >= (maxIterations - 1)) {
		   //lastIteration = 1;
		   cout << "got into last iteration" << endl;
	   }

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
         closestDist = numeric_limits<float>::max();
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
      }
	  

	  //done assigning features to clusters, now we recompute their new centres
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
		  
		  //if a cluster has no members assigned to it
		  if (nMembers[cIdx] <= 0){
			  //leave their centre unchanged
			  //(*newCentroids)[cIdx].content = (*centroids)[cIdx].content;
			  //cout << "cluster " << cIdx << " had 0 members" << endl;
		  }
		  else {
			  //compute their location by division of location with number of members
			  for (size_t dim = 0; dim < dataDims; ++dim) {
				  (*newCentroids)[cIdx].content[dim] /= nMembers[cIdx];
			  }
		  }
      }
      deltaDist = (prevTotalDistance - totalDistance);
      deltaDist = deltaDist < 0 ? deltaDist * -1.0 : deltaDist;
   
   }
   
   ///////////////////////////////////////////////////////

   //now we have the cluster centroids which we consider to be final. 
   //We will consider additional computation in an extra iteration, without recomputing the
   // centroids, but only considering how the features are assigned to them.

   clusters = (*newCentroids);
   unsigned int featclass;
   vector<int> featureAssignments(nData, -1);


   for (size_t cIdx = 0; cIdx < nClusters; ++cIdx) {
	   nMembers[cIdx] = 0;
   }
   //we once more iterate over all features
   for (unsigned int featID = 0; featID < nData; ++featID) {

	   

	   //assigning to cluster:
	   closestDist = numeric_limits<float>::max();
	   unsigned int closestCentr = -1;

	   //compute closest cluster of a feature
	   for (size_t cIdx = 0; cIdx < nClusters; ++cIdx) {
		   curDist = clusters[cIdx].getDistanceSq(featureSamples[featID]);
		   //cout << "curDist = " << curDist << endl;
		   if (curDist < closestDist) {
			   closestDist = curDist;
			   closestCentr = cIdx;
		   }
	   }
	   //now we know closestCentr is the closest cluster, to which feature has a distance of closestDist
	   featclass = featureSamples[featID].getLabelId();
	   featureAssignments[featID] = closestCentr;

	   averageDistances[closestCentr] += closestDist;
	   ++nMembers[closestCentr];

	   byClassAverageDistancesToCentroid[closestCentr][featclass] += closestDist;
	   ++byClassNMembers[closestCentr][featclass];

	   //add to classcluster accumulatively
	   for (size_t dim = 0; dim < dataDims; ++dim) {
		   byClassClusters[closestCentr][featclass].content[dim] += featureSamples[featID].content[dim];
	   }


   }


   //now we can calculate, for every cluster:
   for (unsigned int clustID = 0; clustID < nClusters; ++clustID) {
	   //the average distance of their features
	   averageDistances[clustID] = nMembers[clustID] == 0 ? 0 : averageDistances[clustID] / nMembers[clustID];

	   for (unsigned int classID = 0; classID < nClasses; ++classID) {
		   //coopmuting the centroids of the classclusters
		   //if class has no assignments...
		   if (byClassNMembers[clustID][classID] == 0) {
			   byClassClusters[clustID][classID] = clusters[clustID];
			   byClassAverageDistancesToCentroid[clustID][classID] = 0;
			   byClassClusterDistanceToCentroid[clustID][classID] = 0;
		   }else {
			   for (unsigned int dim = 0;dim < dataDims; ++dim) {
				   byClassClusters[clustID][classID].content[dim] /= byClassNMembers[clustID][classID];
			   }
			   //and for every class the average distance of its features to the centroid 
			   byClassAverageDistancesToCentroid[clustID][classID] /= byClassNMembers[clustID][classID];
			   //and the average distance of the classcluster to the centroid:
			   int count = 0;
			   for (unsigned int dim = 0; dim < dataDims; ++dim) {
				   if (clusters[clustID].content[dim] == byClassClusters[clustID][classID].content[dim]) {
					   ++count;
				   }
			   }
			   byClassClusterDistanceToCentroid[clustID][classID] = clusters[clustID].getDistanceSq(byClassClusters[clustID][classID]); //.getDistanceSq(clusters[clustID]);
		   }
		}
   }

   cout << "third flag!" << endl;

   //now, in order to compute their deviations, we must iterate over the features ones more. 
   // fortunately, we won't have to recompute all distances, because we saved the assignments previously. 
   unsigned int clusterAssigned = 0;
   float featDistClust;
   for (unsigned int featID = 0; featID < nData; ++featID) {
	   clusterAssigned = featureAssignments[featID];
	   
		
	   //now we know closestCentr is the closest cluster, to which feature has a distance of closestDist
	   featclass = featureSamples[featID].getLabelId();

	   featDistClust = (clusters[clusterAssigned].getDistanceSq(featureSamples[featID]) - averageDistances[clusterAssigned]) * (clusters[clusterAssigned].getDistanceSq(featureSamples[featID]) - averageDistances[clusterAssigned]);
	   deviations[clusterAssigned] += featDistClust;
	   byClassDeviationsToCentroid[clusterAssigned][featclass] += featDistClust;
	   byClassAverageDistancesToClassCluster[clusterAssigned][featclass] += byClassClusters[clusterAssigned][featclass].getDistanceSq(featureSamples[featID]);


   }

   // now we can compute the accumulative deviation within a cluster, deviations per class to cluster, and for every class average distance to classcluster
   
   for (unsigned int clustID = 0; clustID < nClusters; ++clustID) {
	   //the average distance of their features
	   deviations[clustID]  = nMembers[clustID] == 0 ? 0 : deviations[clustID] / nMembers[clustID];

	   for (unsigned int classID = 0; classID < nClasses; ++classID) {
		   
		   byClassDeviationsToCentroid[clustID][classID] = byClassNMembers[clustID][classID] == 0 ? 0 :  byClassDeviationsToCentroid[clustID][classID] / byClassNMembers[clustID][classID];
		   byClassAverageDistancesToClassCluster[clustID][classID] = byClassNMembers[clustID][classID]  == 0 ? 0 : byClassAverageDistancesToClassCluster[clustID][classID] / byClassNMembers[clustID][classID];
		   byClassContributions[clustID][classID] = byClassNMembers[clustID][classID] / nMembers[clustID];
		   
	   }
   }

 

   
   clusters = (*newCentroids);
   clusterByClassContributions = byClassContributions;
   
   //print everything
   printAllClusterStats();



   return (*newCentroids);
}

vector<vector<float> > AKMeans::getClusterClassContributions() {
	return clusterByClassContributions;
}

vector<float>  AKMeans::getClusterClassContributions(int clust) {
	return clusterByClassContributions[clust];
}

vector<float>  AKMeans::getClusterClassContributions(Feature feat) {

	float curDist;
	/* float prevTotalDistance = 2;
	float totalDistance = 1;	//used to track cluster changes in mean
	float deltaDist = 1;
	*/
	float closestDist;
	//unsigned int dataDims = feat.content.size();	//feature dimensionality
	unsigned int nClusters = clusters.size();
	closestDist = numeric_limits<float>::max();
	unsigned int closestCentr = -1;

	//compute closest cluster of a feature
	for (size_t cIdx = 0; cIdx < nClusters; ++cIdx) {
		curDist = clusters[cIdx].getDistanceSq(feat);
		//cout << "curDist = " << curDist << endl;
		if (curDist < closestDist) {
			closestDist = curDist;
			closestCentr = cIdx;
		}
	}
	
	return clusterByClassContributions[closestCentr];
}

void AKMeans::printAllClusterStats() {
	string sep(" | ");
	for (unsigned int clustID = 0; clustID < nClusters; ++clustID) {
		cout << "cluster: " << clustID << ":" << endl;
		cout << "class:" << " | " <<
			"# members" << " | " <<
			"avgdist to centr" << " | " <<
			"std to centr" << " | " <<
			"avgdist to classclust" << sep <<
			"dist(centr,classclust)" << sep << 
			"contributions(prop)"	<< sep << endl;
		cout << setw(6) << "all" << sep <<
			setw(9) << nMembers[clustID] << sep <<
			setw(16) << averageDistances[clustID] << sep <<
			setw(12) << deviations[clustID] << sep <<
			endl;

		
		for (unsigned int classID = 0; classID < nClasses; ++classID) {
			cout << 
				setw(6) << classID << sep <<
				setw(9) << byClassNMembers[clustID][classID] << sep <<
				setw(16) << byClassAverageDistancesToCentroid[clustID][classID] << sep <<
				setw(13) << byClassDeviationsToCentroid[clustID][classID] << sep <<
				setw(21) << byClassAverageDistancesToClassCluster[clustID][classID] << sep << 
				setw(22) << byClassClusterDistanceToCentroid[clustID][classID]<< sep <<
				setw(19) << clusterByClassContributions[clustID][classID] << sep << endl;
		}
	}
}

void AKMeans::printClusterStats(unsigned int clustID) {
	string sep(" | ");
	cout << "cluster: " << clustID << ":" << endl;
	cout << "class:" << " | " <<
		"# members" << " | " <<
		"avgdist to centr" << " | " <<
		"std to centr" << " | " <<
		"avgdist to classclust" << sep <<
		"dist(centr,classclust)" << sep << endl;
	cout << setw(6) << "all" << sep <<
		setw(9) << nMembers[clustID] << sep <<
		setw(16) << averageDistances[clustID] << sep <<
		setw(12) << deviations[clustID] << sep <<
		endl;


	for (unsigned int classID = 0; classID < nClasses; ++classID) {
		cout <<
			setw(6) << classID << sep <<
			setw(9) << byClassNMembers[clustID][classID] << sep <<
			setw(16) << byClassAverageDistancesToCentroid[clustID][classID] << sep <<
			setw(13) << byClassDeviationsToCentroid[clustID][classID] << sep <<
			setw(21) << byClassAverageDistancesToClassCluster[clustID][classID] << sep <<
			setw(22) << byClassClusterDistanceToCentroid[clustID][classID] << sep <<
			endl;
	}
}


