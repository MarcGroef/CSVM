#include <csvm/csvm_lvq.h>
#include <cstdio>
using namespace std;
using namespace csvm;

LVQ::LVQ(){
   
   srand(time(NULL));
}


void checkEquals(vector< Feature> dictionary){
   unsigned int dictSize = dictionary.size();
   int nEquals = 0;
   bool isEqual;
   unsigned int wordSize = dictionary[0].content.size();;
   
   for(size_t word = 0; word < dictSize; ++word){
      
      for(size_t word1 = word; word1 < dictSize; ++word1){
         if(word1==word) continue;
         isEqual = true;
         for(size_t d = 0; d < wordSize; ++d){
            if(dictionary[word].content[d] != dictionary[word1].content[d])
               isEqual = false;
         }
         //if(isEqual)
           // cout << "cluster " << word << " and " << word1 << " are equal\n";
         if (isEqual)
         ++nEquals;
      }
      
   }
   cout << "I found " << nEquals << " equal words, out of " << dictSize << " words\n";
   
}
//this goes horribly wrong! 
vector<Feature> LVQ::initPrototypes(vector<Feature> collection, unsigned int labelId, unsigned int nProtos){
   
   //dictionary.reserve(nProtos);
   int collectionSize = collection.size();
   unsigned int featureSize = collection[0].content.size();
   vector<Feature> dictionary(nProtos,Feature(featureSize, 0));
   
  
      
   unsigned int randomInt;
   //cout <<"initProto's\n";
   
   for(size_t idx = 0; idx < nProtos; ++idx){
      randomInt = rand() % collectionSize;
      
      while(randomInt < 0) randomInt += collectionSize;
      
      //f = &collection[randomInt];
      for(size_t d = 0; d < collection[randomInt].content.size(); ++d){
         
         dictionary[idx].content[d] = collection[randomInt].content[d];
        // cout << newFeat.content[d] << endl;
      }
      //dictionary.push_back(newFeat);
   }
   //getchar();
   checkEquals(dictionary);
   return dictionary;
}

vector<Feature> LVQ::cluster(vector<Feature> collection, unsigned int labelId, unsigned int numberPrototypes, double learningRate, int epochs){
   vector<Feature> dictionary;
   unsigned int collectionSize = collection.size();
   vector<double> distances(numberPrototypes, 0);
   unsigned int featureDims = collection[0].size;
   dictionary = initPrototypes(collection, labelId, numberPrototypes);
   cout << "sanity\n";
   checkEquals(dictionary);
   
   double minDist;
   unsigned int closestProto;
   
   for(int epoch = 0; epoch < epochs ; ++epoch){
      //cout << "Epoch " << epoch << "\n";
      for(size_t idx = 0; idx < collectionSize; ++idx){  //loop through datapoints
         
         minDist = 9999999999;
         closestProto = 0;
         //calc distances and closest prototype
         for(size_t proto = 0; proto < numberPrototypes; ++proto){
            //cout << "proto[0]: " << dictionary[proto].content[0] << endl;
            //for(size_t d = 0; d < dictionary[proto].content.size(); ++d)
            distances[proto] = dictionary[proto].getDistanceSq( &collection[idx]);
            //cout << "measured distances = " << distances[proto] << endl;
            
            if(distances[proto] < minDist){
               minDist = distances[proto];
               closestProto = proto;
            }
         }
         //if(idx > 0) cout << dictionary[idx].getDistanceSq(&dictionary[idx - 1]) << endl;
        // if(idx > 0) cout << (&dictionary[idx].content[0] == &dictionary[idx - 1].content[0] ? "Same ptr" : "diff ptr") << endl;
         //update prototype
         
         //move closest proto towards vector
         for(size_t dim = 0; dim < featureDims; ++dim){
            dictionary[closestProto].content[dim] -= learningRate * (collection[idx].content[dim] - dictionary[closestProto].content[dim]);               
         }
         
         //move other prototypes away from vector
         for(size_t proto = 0; proto < numberPrototypes; ++proto){
            if(proto != closestProto){
               for(size_t dim = 0; dim < featureDims; ++dim){
                  dictionary[proto].content[dim] += learningRate * (collection[idx].content[dim] - dictionary[proto].content[dim]);     
               }
            }
         }
         
         
         
      }
   }
   checkEquals(dictionary);
   getchar();
   return dictionary;
}