#include <csvm/csvm_lvq.h>

using namespace std;
using namespace csvm;

LVQ::LVQ(){
   
   srand(time(NULL));
}

vector<Feature> LVQ::initPrototypes(vector<Feature> collection, unsigned int labelId, unsigned int nProtos){
   vector<Feature> dictionary;
   dictionary.reserve(nProtos);
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   Feature* f;
   Feature newFeat(featureSize, 0);
   for(size_t idx = 0; idx < nProtos; ++idx){
      f = &collection[rand() % collectionSize];
      for(size_t d = 0; d < f->content.size(); ++d){
         newFeat.content[d] = f->content[d];
        // cout << newFeat.content[d] << endl;
      }
      dictionary.push_back(newFeat);
   }
   return dictionary;
}

vector<Feature> LVQ::cluster(vector<Feature> collection, unsigned int labelId, unsigned int numberPrototypes, double learningRate, int epochs){
   vector<Feature> dictionary;
   unsigned int collectionSize = collection.size();
   vector<double> distances(numberPrototypes, 0);
   unsigned int featureDims = collection[0].size;
   dictionary = initPrototypes(collection, labelId, numberPrototypes);
   
   double minDist;
   unsigned int closestProto;
   for(int epoch = 0; epoch < epochs ; ++epoch){
      //cout << "Epoch " << epoch << "\n";
      for(size_t idx = 0; idx < collectionSize; ++idx){  //loop through datapoints
         Feature* f = &collection[idx];
         minDist = 9999999999;
         closestProto = 0;
         //calc distances and closest prototype
         for(size_t proto = 0; proto < numberPrototypes; ++proto){
            //cout << "proto[0]: " << dictionary[proto].content[0] << endl;
            //for(size_t d = 0; d < dictionary[proto].content.size(); ++d)
            distances[proto] = dictionary[proto].getDistanceSq(f);
            //cout << "measured distances = " << distances[proto] << endl;
            
            if(distances[proto] < minDist){
               minDist = distances[proto];
               closestProto = proto;
            }
         }
         
         //update prototype
         
         //move closest proto towards vector
         for(size_t dim = 0; dim < featureDims; ++dim){
            dictionary[closestProto].content[dim] -= learningRate * (f->content[dim] - dictionary[closestProto].content[dim]);
               
         }
         
         //move other prototypes away from vector
         for(size_t proto = 0; proto < numberPrototypes; ++proto){
            if(proto != closestProto){
               for(size_t dim = 0; dim < featureDims; ++dim){
                  dictionary[proto].content[dim] += learningRate * (f->content[dim] - dictionary[proto].content[dim]);
                     
               }
            }
         }
         
         
         
      }
   }
   
   return dictionary;
}