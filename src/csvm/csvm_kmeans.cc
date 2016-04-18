#include <csvm/csvm_kmeans.h>
//#include <stdlib.h>

/*  A class implementing KMeans.
 * 
 * 
 * 
 * */

using namespace std;
using namespace csvm;

void KMeans::setSettings(KMeans_settings s){
   settings = s;
}


//Initialize centroids by giving them the location of K-different datapoints in the collection

vector<Centroid> KMeans::initCentroids(vector<Feature> collection, unsigned int nClusters){
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   vector<Centroid> dictionary(nClusters);
   
   for(size_t clIdx = 0; clIdx < nClusters; ++clIdx){
      dictionary[clIdx].content.resize(featureSize);
   }
   
      
   unsigned int randomInt;
   
   for(size_t idx = 0; idx < nClusters; ++idx){
      
      randomInt = rand() % collectionSize;
      
      for(size_t d = 0; d < featureSize; ++d){
         
			double randDouble = (((double)rand() / 1000) / RAND_MAX );
			//randDouble -= randDouble / 2;
			
         dictionary[idx].content[d] = collection[randomInt].content[d] + randDouble;
      }
   }
   
   return dictionary;

}


//The actual KMeans clusterings
vector<Centroid> KMeans::cluster(vector<Feature>& featureSamples, unsigned int nClusters){
   unsigned int nData = featureSamples.size(); 
	//location of 2-times the centroids. One for currect centroid location, and one for the next iteration.
	//pointers 'centroids' and 'newCentroids' denote them, and are updated at the end of each iteration to point to the correct one.
	//This minimalizes memory hassle.
	
   vector<Centroid> centroids0 = initCentroids(featureSamples, nClusters);
   vector<Centroid> centroids1(nClusters);
   
   
   vector<Centroid>* centroids = &centroids0;
   vector<Centroid>* newCentroids = &centroids1;
	
   int curCentroids = 1;
   unsigned int dataDims = centroids0[0].content.size();
   
   
   vector< unsigned int > nMembers(nClusters,0);
 
   
   double curDist;
   double prevTotalDistance = 2;
   double totalDistance = 1;
   double deltaDist = 1;
   double closestDist;
   
   for(size_t clIdx = 0; clIdx < nClusters; ++clIdx){
      centroids1[clIdx].content.resize(dataDims);
   }

   ofstream statFile;
   if(settings.liveROut){
    statFile.open( "CL_DIST.csv" );
    system("Rscript ../genRplotsLive CL_DIST &> ./logs/errorLOG");
    statFile << "Iteration,Distance" << endl;
   }

   for(size_t itx = 0; /*deltaDist > 0*/ itx < settings.nIter; curCentroids *= -1, ++itx){
//      if(settings.debugOut)         cout << "KMeans iteration " << itx << "/" << settings.nIter << endl;
      prevTotalDistance = totalDistance;
      totalDistance = 0.0;
      
      centroids = (curCentroids == 1 ? &centroids0: &centroids1);
      newCentroids = (curCentroids == 1 ? &centroids1 : &centroids0);
      
		//reset the centroids: set content to zero, and reset memberships
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
         for(size_t dim = 0; dim < dataDims; ++dim)
            (*newCentroids)[cIdx].content[dim] = 0.0;
         nMembers[cIdx] = 0;
      }
         
      //for all data, determine the nearest centroid
      for(size_t dIdx = 0; dIdx < nData; ++dIdx){
         closestDist = (*centroids)[0].getDistanceSq(featureSamples[dIdx]);
         unsigned int closestCentr = 0;
         
         for(size_t cIdx = 1; cIdx < nClusters; ++cIdx){
            curDist = (*centroids)[cIdx].getDistanceSq(featureSamples[dIdx]);
            //cout << "curDist = " << curDist << ", min dist: " << closestDist << endl;
            if(curDist < closestDist){
               closestDist = curDist;
               closestCentr = cIdx;
               
            }
         }
         //notate total Dist, as a nice statistic, that doesnt really say anything
         totalDistance += closestDist;
         
         for(size_t dim = 0; dim < dataDims; ++dim)
            (*newCentroids)[closestCentr].content[dim] += featureSamples[dIdx].content[dim];
         ++nMembers[closestCentr];
      }
//      if(settings.debugOut)cout << "Total distance = " << prevTotalDistance << endl;
      //move to mean position of members
      for(size_t cIdx = 0; cIdx < nClusters; ++cIdx){
        // cout << cIdx << " has " << nMembers[cIdx] << " members!! @ iter "<< itx <<" \n";
         if(nMembers[cIdx] > 0)
            for(size_t dim = 0; dim < dataDims; ++dim)
               (*newCentroids)[cIdx].content[dim] /= nMembers[cIdx];
         else{ //keep it at current position, if no new members
            for(size_t dim = 0; dim < dataDims; ++dim)
               (*newCentroids)[cIdx].content[dim] = (*centroids)[cIdx].content[dim];
           cout << cIdx << " has no members!! @ iter "<< itx <<" \n";
         }
      }
      deltaDist = (prevTotalDistance - totalDistance);
      deltaDist = deltaDist < 0 ? deltaDist * -1.0 : deltaDist;

      if(settings.liveROut)
        statFile << itx << "," << totalDistance << endl;
   }

   if(settings.liveROut){
     statFile << "EOF" << endl;   
     statFile.close();
   }
	//tadaa, fresh made centroids
   return (*newCentroids);
}




