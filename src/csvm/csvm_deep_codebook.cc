#include <csvm/csvm_deep_codebook.h>

using namespace std;
using namespace csvm;



void DeepCodebook::setSettings(DeepCodebookSettings s){
   this->settings = s;
   layerStack.resize(settings.nHiddenLayers);
}

vector<double> DeepCodebook::flowUpUntil(vector<double>& imagePatchActivations){
   return vector<double>();
}

void DeepCodebook::constructPatchLayer(vector<Feature>& patchCollection){
   layerStack[0] = kmeans.cluster(patchCollection, settings.nPatchCentroids);
}

void DeepCodebook::constructHiddenLayers(vector< vector<double> >& imagePatchActivations){
   vector<Feature> activations;
   
   layerStack[1] = kmeans.cluster(imagePatchActivations, settings.layerSizes[0]);
   imagePatchActivations.clear();
   
   activations = 
   for(size_t layerIdx = 0; layerIdx < settings.nHiddenLayers; ++layerIdx){
      layerStack[1 + layerIdx] = kmeans.cluster()
   }
}

Feature DeepCodebook::getActivations(vector<Feature>& imagePaches){
   return imagePaches[0];
}