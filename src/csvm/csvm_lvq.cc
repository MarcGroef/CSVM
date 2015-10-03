#include <csvm/csvm_lvq.h>

using namespace std;
using namespace csvm;

LVQ::LVQ(){
   
   srand(time(NULL));
}

vector<Feature> LVQ::initPrototypes(vector<Feature> collection, unsigned int nProtos){
   vector<Feature> dictionary;
   dictionary.reserve(nProtos);
   int collectionSize = collection.size();
   for(size_t idx = 0; idx < nProtos; ++idx)
      dictionary.push_back(collection[rand() % collectionSize]);
   return dictionary;
}

vector<Feature> LVQ::cluster(vector<Feature> collection, unsigned int numberPrototypes, double learningRate){
   vector<Feature> dictionary;
   unsigned int collectionSize = collection.size();
   double distances[numberPrototypes];
   unsigned int featureDims = collection[0].size;
   dictionary = initPrototypes(collection,numberPrototypes);
   
   double minDist;
   int closestProto;
   
   for(size_t idx = 0; idx < collectionSize; ++idx){  //loop through datapoints
      Feature* f = &collection[idx];
      minDist = 9999999999;
      
      //calc distances and closest prototype
      for(size_t proto = 0; proto < numberPrototypes; ++proto){
         
         distances[proto] = 0;
         
         for(size_t dim = 0; dim < featureDims; ++dim){
            //squared distance
            distances[proto] += (dictionary[proto].content[dim] - f->content[dim]) * (dictionary[proto].content[dim] - f->content[dim]);
         }
         if(distances[proto] < minDist){
            minDist = distances[proto];
            closestProto = proto;
         }
      }
      
      //update prototype
      
      for(size_t dim = 0; dim < featureDims; ++dim){
         dictionary[closestProto].content[dim] += learningRate * (f->content[dim] - dictionary[closestProto].content[dim]);
            
      }
      
      
      
   }
   
   return dictionary;
}