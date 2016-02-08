#include <csvm/csvm_lvq.h>
#include <cstdio>

//DEPRECATED

using namespace std;
using namespace csvm;

LVQ::LVQ(){
   
   srand(time(NULL));
}


vector<Feature> LVQ::initPrototypes(vector<Feature> collection, unsigned int labelId, unsigned int nProtos){
   //cout << "Initializing centroids..\n";
   //dictionary.reserve(nProtos);
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   vector<Feature> dictionary(nProtos,Feature(featureSize, 0));
   
   //getchar();
      
   unsigned int randomInt;
   //cout <<"initProto's\n";
   
   for(size_t idx = 0; idx < nProtos; ++idx){
      randomInt = rand() % collectionSize;
      
      while(randomInt < 0) randomInt += collectionSize;
      
      //f = &collection[randomInt];
      for(size_t d = 0; d < collection[0].content.size(); ++d){
         
         dictionary[idx].content[d] = collection[randomInt].content[d];
        // cout << newFeat.content[d] << endl;
      }
      //dictionary.push_back(newFeat);
   }
   //getchar();
   //checkEquals(dictionary);
   //cout << "Done initializing centroids.\n";
   return dictionary;
}

void printCollection(vector<Feature> vec){
   size_t nFeatures = vec.size();
   size_t dims = vec[0].content.size();
   
   for(size_t feat = 0; feat < nFeatures; ++feat){
      cout << "Word " << feat << ":";
      for(size_t d = 0; d < dims; ++d){
         cout << vec[feat].content[d] << ", ";
      }
      cout << endl;
   }
   
}

vector<Feature> LVQ::cluster(vector<Feature> collection, unsigned int labelId, unsigned int numberPrototypes, double learningRate, int epochs){
   vector<Feature> dictionary;
   unsigned int collectionSize = collection.size();
   
   unsigned int featureDims = collection[0].size;
   
   dictionary = initPrototypes(collection, labelId, numberPrototypes);
   //cout << "sanity\n";
   //checkEquals(dictionary);
   //printCollection(dictionary);
   //getchar();
   double minDist;
   unsigned int closestProto;
   double distance;
   for(int epoch = 0; epoch < epochs ; ++epoch){
      //cout << "Epoch " << epoch << "\n";
      for(size_t idx = 0; idx < collectionSize; ++idx){  //loop through datapoints
         
         minDist = sqrt(dictionary[0].getDistanceSq( collection[idx]));
         closestProto = 0;
         //calc distances and closest prototype
         for(size_t proto = 1; proto < numberPrototypes; ++proto){
            
            distance = sqrt(dictionary[proto].getDistanceSq( collection[idx]));
            //cout << "dists: proto" << proto << ": " << distance << endl;
            
            if(distance < minDist){
               minDist = distance;
               closestProto = proto;
            }
         }
         //cout << "closestProto= " << closestProto << endl;
         for(size_t dim = 0; dim < featureDims; ++dim){
            
            dictionary[closestProto].content[dim] += learningRate * (collection[idx].content[dim] - dictionary[closestProto].content[dim]);               
         }
         
         //move other prototypes away from vector
         /*for(size_t proto = 0; proto < numberPrototypes; ++proto){
            if(proto != closestProto){
               for(size_t dim = 0; dim < featureDims; ++dim){
                  dictionary[proto].content[dim] -= learningRate * (collection[idx].content[dim] - dictionary[proto].content[dim]);     
               }
            }
         }*/
         
         
         
      }
      //cout << "Epoch evaluation:\n";
      //checkEquals(dictionary);
      //getchar();
   }
  
   return dictionary;
}