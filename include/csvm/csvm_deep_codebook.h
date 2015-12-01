#ifndef CSVM_DEEP_CODEBOOK_H
#define CSVM_DEEP_CODEBOOK_H

#include <vector>
#include <cmath>

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
      KMeans kmeans;
      
      Feature flowUpUntil(vector< vector<Feature> >& imagePatches, unsigned int untilThisLayer);
      double getCentroidActivation(Centroid& centroid, Feature& f);
      Feature getPatchActivations(vector< vector<Feature> >& imagePatches);
   public:
      void setSettings(DeepCodebookSettings* s);
      void constructPatchLayer(vector<Feature>& patchCollection);
      void constructHiddenLayers(vector< vector< vector<Feature> > >& imagePatches);
      Feature getActivations(vector< vector<Feature> >& imagePatches);
   };
   
   
}
#endif