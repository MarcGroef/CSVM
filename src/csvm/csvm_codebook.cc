#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodebook(vector<Feature> featureset){
   settings.method = LVQ_Clustering;
   settings.numberVisualWords = 100;
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, 100, 0.02,10);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset);
         break;
   }
   
}

Feature Codebook::getActivations(Feature* f){
   Feature act(settings.numberVisualWords,0);
   double distances[settings.numberVisualWords];
   double meanDist = 0;
   double dev;
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
      distances[word] = bow[word].getDistanceSq(f);
      meanDist += distances[word];
   }
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
      dev = meanDist - distances[word];
      act.content[word] = dev > 0 ? dev : 0;
   }
   
   
   
   return act;
}