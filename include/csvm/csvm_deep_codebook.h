#ifndef CSVM_DEEP_CODEBOOK_H
#define CSVM_DEEP_CODEBOOK_H

#include <vector>

#include "csvm_feature.h"
#include "csvm_patch.h"
#include "csvm_centroid.h"

#include "csvm_kmeans.h"

using namespace std;

namespace csvm{
   
   struct DeepCodebookSettings{
     unsigned int nHiddenLayers;
     vector<unsigned int> layerSizes;
     unsigned int nPatchCentroids;
      
   };
   
   class DeepCodebook{
      DeepCodebookSettings settings;
      vector< vector< Centroid > > layerStack;
      vector<double> flowUpUntil(vector<double>& imagePatchActivations);
      KMeans kmeans;
      
   public:
      void setSettings(DeepCodebookSettings s);
      void constructPatchLayer(vector<Feature>& patchCollection);
      void constructHiddenLayers(vector< vector<double> >& imagePatchActivations);
      Feature getActivations(vector<Feature>& imagePaches);
   };
   
   
}
#endif