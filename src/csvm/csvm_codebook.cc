#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodebook(vector<Feature> featureset){
   settings.method = LVQ_Clustering;
   settings.numberVisualWords = 100;
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, settings.numberVisualWords, 0.02,10);
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
   
   act.label = f->label;
   
   return act;
}

void Codebook::exportCodebook(string filename){
   /* codebook file conventions:
      first one line with one number, representing the number of visual words
      seconds, one line with one number: the number of bytes of each visual words.
      third a little-endian binary dump of the visual words.
   */
   
   
   
   
   
}