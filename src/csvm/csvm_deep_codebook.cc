#include <csvm/csvm_deep_codebook.h>

using namespace std;
using namespace csvm;

DeepCodebook::DeepCodebook(){
   settings.nHiddenLayers  = 3;
   settings.layerSizes.push_back(100);
   settings.layerSizes.push_back(100);
   settings.nPatchCentroids = 100;
   
   KMeans_settings mSets;
   mSets.nClusters = 0;
   
   mSets.alpha = 0;
   mSets.nIter = 15;
   kmeans.setSettings(mSets);
   
   layerStack.resize(settings.nHiddenLayers);
}

void DeepCodebook::setSettings(DeepCodebookSettings* s){
   this->settings = *s;
   layerStack.resize(settings.nHiddenLayers);
   
   
}

Feature DeepCodebook::flowUpUntil(vector< vector<Feature> >& imagePatches, unsigned int untilThisLayer){
   Feature activations = getPatchActivations(imagePatches);
   

   
   for(size_t lIdx = 1; lIdx < untilThisLayer; ++lIdx){
      Feature newActivations(settings.layerSizes[lIdx],0);
      for(size_t centrIdx = 0; centrIdx < settings.layerSizes[lIdx]; ++centrIdx){
         newActivations.content[centrIdx] = getCentroidActivation(layerStack[lIdx][centrIdx],activations);
      }
      activations.content.clear();
      activations = newActivations;
      newActivations.content.clear();
   }
   return activations;
}

void DeepCodebook::constructPatchLayer(vector<Feature>& patchCollection){
   cout << "constructing patch layer\n"; 
   vector<Centroid> km = kmeans.cluster(patchCollection, settings.nPatchCentroids);
   cout << "asigned patch layer\n";
   layerStack[0] = km;
   
}

void DeepCodebook::constructHiddenLayers(vector< vector< vector<Feature> > >& imagePatches){
   unsigned int nImages = imagePatches.size();
   cout << "Constructing hidden layers.\n";
   //activations = 
   for(size_t layerIdx = 1; layerIdx < settings.nHiddenLayers + 1; ++layerIdx){
      cout << "constructing layer " << layerIdx << endl;
      vector<Feature> activations;
      for(size_t imIdx = 0; imIdx < nImages;++imIdx){
         activations.push_back(flowUpUntil(imagePatches[imIdx], layerIdx));
      }
      layerStack[layerIdx] = (kmeans.cluster(activations, settings.layerSizes[layerIdx - 1]));
   }
}

double DeepCodebook::getCentroidActivation(Centroid& c, Feature& f){
   double xx = 0;
   double cc = 0;
   double xc = 0;
   unsigned int dataDims = c.content.size();
   
   
   for(size_t dIdx = 0; dIdx < dataDims; ++dIdx){
      xx += f.content[dIdx] * f.content[dIdx];
      cc += c.content[dIdx] * c.content[dIdx];
      xc += c.content[dIdx] * f.content[dIdx];
   } 
   return sqrt(cc + xx - (2 * xc));
}

Feature DeepCodebook::getPatchActivations(vector< vector<Feature> >& imagePatches){
   vector<double> finalFeature;
   unsigned int nQuadrants = 4;
   
   Feature activations(settings.nPatchCentroids * nQuadrants,0.0);
   
   for(size_t qIdx = 0;qIdx < nQuadrants; ++qIdx){
      unsigned int nPatches = imagePatches[qIdx].size();
      
      for(size_t fIdx = 0; fIdx < nPatches; ++fIdx){
         for(size_t centrIdx = 0; centrIdx < settings.nPatchCentroids; ++centrIdx){
            activations.content[qIdx * settings.nPatchCentroids + centrIdx] = getCentroidActivation(layerStack[0][centrIdx], imagePatches[qIdx][fIdx]);
         }
      }
   }
   return activations;
}

Feature DeepCodebook::getActivations(vector< vector<Feature> >& imagePatches){
   return flowUpUntil(imagePatches, settings.nHiddenLayers);
}

