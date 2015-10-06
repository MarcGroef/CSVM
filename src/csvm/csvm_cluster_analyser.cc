#include <csvm/csvm_cluster_analyser.h>

using namespace std;
using namespace csvm;

ClusterAnalyser::ClusterAnalyser(){
   
   
}

void ClusterAnalyser::setSettings(ClusterAnalyserSettings set){
   settings = set;
   switch(settings.method){
     case CSVM_RBM:
       rbm.setSettings(settings.rbmSettings);  //TODO: 
       break;
     
   }
}

void ClusterAnalyser::studyFeatures(vector<Feature> features){
   unsigned int nFeatures = features.size();
   unsigned int featureDim = features[0].content.size();
   cout << "Studying " << nFeatures << " 1 x " << featureDim << " dimensional features\n";
   double** data;
   
   
   data = (double**) malloc(nFeatures * sizeof(double*));
   assert(data!=NULL);
   
   for(size_t idx = 0; idx < nFeatures; ++ idx){
      data[idx] = (double*) malloc(featureDim * sizeof(double));
      assert(data[idx] != NULL);
      for(size_t idx1 = 0; idx1 < featureDim; ++idx1){
         data[idx][idx1] = features[idx].content[idx1];
      }
   }
   features.clear();
   cout << "Created RBM dataset with " << nFeatures << " 1 x " << featureDim << " features\n";
   rbm.linkDataset(data,nFeatures);
   rbm.train();
   rbm.freeDataset();
}

ClusterAnalyser::~ClusterAnalyser(){
   
}