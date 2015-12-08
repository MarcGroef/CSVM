#ifndef CSVM_DCB_CONV_LAYER_H
#define CSVM_DCB_CONV_LAYER_H

#include "csvm_dcb_featuremap.h"
#include "csvm_dcb_poolmap.h"

#include <vector>

using namespace std;



namespace csvm{
   
   class ConvLayer{
     vector<FeatureMap> featureMaps;
     vector<PoolMap> poolMaps;
     unsigned int scanSize;
     unsigned int nCentroids;
     unsigned int fmHeight, fmWidth;
     unsigned int poolHeight, poolWidth;
   public:
      ConvLayer(unsigned int nCentroids, unsigned int fmWidth, unsigned int fmHeight, unsigned int poolWidth, unsigned int poolHeight);
      void poolFeatureMaps();
   };

}

#endif