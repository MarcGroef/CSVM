#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodebook(vector<Feature> featureset){
   settings.method = LVQ_Clustering;
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, 100, 0.02,10);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset);
         break;
   }
   
}