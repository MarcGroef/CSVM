#include <csvm/csvm_kmeans.h>
//#include <stdlib.h>


using namespace std;
using namespace csvm;

void KMeans::setSettings(KMeans_settings s){
   settings = s;
}

vector<Centroid> KMeans::initCentroids(vector<Feature> collection, unsigned int nClusters){
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   vector<Centroid> dictionary(nClusters);
   
   for(size_t clIdx = 0; clIdx < nClusters; ++clIdx){
      dictionary[clIdx].content.resize(featureSize);
   }
   
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



vector<Centroid> KMeans::cluster(vector<Feature>& featureSamples, unsigned int nClusters){
   cout << "nIter =" << settings.nIter << endl;
   unsigned int nData = featureSamples.size(); 
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
   
   
   cout << "nIter =" << settings.nIter << endl;
   for(size_t itx = 0; /*deltaDist > 0*/ itx < settings.nIter; curCentroids *= -1, ++itx){
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
   cout << "yay, kmena sis done\n";
   return (*newCentroids);
}




