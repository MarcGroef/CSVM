#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodeBook(vector<Feature> featureset){
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, 100, 0.02);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset);
         break;
   }
   
}