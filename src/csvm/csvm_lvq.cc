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

vector<Feature> LVQ::cluster(vector<Feature> collection, unsigned int numberPrototypes, double learningRate, int epochs){
   vector<Feature> dictionary;
   unsigned int collectionSize = collection.size();
   vector<double> distances(numberPrototypes);
   unsigned int featureDims = collection[0].size;
   dictionary = initPrototypes(collection,numberPrototypes);
   
   double minDist;
   int closestProto;
   for(int epoch = 0; epoch < epochs ; ++epoch){
      cout << "Epoch " << epoch << "\n";
      for(size_t idx = 0; idx < collectionSize; ++idx){  //loop through datapoints
         Feature* f = &collection[idx];
         minDist = 9999999999;
         
         //calc distances and closest prototype
         for(size_t proto = 0; proto < numberPrototypes; ++proto){
            
            
            distances[proto] = dictionary[proto].getDistanceSq(f);
            
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
   }
   
   return dictionary;
}