#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodeBook(vector<Feature> featureset){
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset);
         break;
   }
   
}