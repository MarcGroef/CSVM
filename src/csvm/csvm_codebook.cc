#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodebook(vector<Feature> featureset){
   settings.method = /*KMeans_Clustering;*/LVQ_Clustering;
   settings.numberVisualWords = 300;
   cout << "clustering " << featureset.size() << " 1 x " << featureset[0].content.size() <<" features..\n";
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, settings.numberVisualWords, 0.05,10);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset, 8);
         break;
   }
   
}

Feature Codebook::getActivations(Feature* f){
   Feature act(settings.numberVisualWords,0);
   vector<double> distances(settings.numberVisualWords);
   double meanDist = 0;
   double dev;
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
//       distances[word] = sqrt(bow[word].getDistanceSq(f));
      distances[word] = bow[word].getManhDist(f);
      meanDist += distances[word];
   }
   meanDist /= (double)settings.numberVisualWords;
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
      //dev = meanDist - distances[word];
      dev = 1 - distances[word] / meanDist;
      act.content[word] = dev > 0 ? dev : 0;
      //out << act.content[word] << endl;
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