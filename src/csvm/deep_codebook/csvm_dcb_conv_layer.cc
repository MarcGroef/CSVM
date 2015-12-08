#include <csvm/deep_codebook/csvm_dcb_conv_layer.h>

using namespace std;
using namespace csvm;

ConvLayer::ConvLayer(unsigned int nCentroids, unsigned int fmWidth, unsigned int fmHeight, unsigned int poolWidth, unsigned int poolHeight){
   featureMaps = vector<FeatureMap>(nCentroids,FeatureMap(fmWidth, fmHeight));
   poolMaps = vector<PoolMap>(nCentroids, PoolMap(poolWidth, poolHeight));
   this->nCentroids = nCentroids;
   this->poolWidth = poolWidth;
   this->poolHeight = poolHeight;
   this->fmWidth = fmWidth;
   this->fmHeight = fmHeight;
   
   if(fmWidth % poolWidth != 0 || fmHeight % poolHeight != 0){
      cout << "ConvLayer Constructor: poolmap doesnt allign to featuremap!\n";
   }
}

void ConvLayer::poolFeatureMaps(){
   unsigned int scanHeight = fmHeight / poolHeight;
   unsigned int scanWidth = fmWidth / poolWidth;
   
   for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
      for(size_t poolX = 0; poolX < poolWidth; ++poolX){
         for(size_t poolY = 0; poolY < poolHeight; ++poolY){
            double sum = 0;
            for(size_t scanX = 0; scanX < scanWidth; ++scanX){
               for(size_t scanY = 0; scanY < scanHeight; ++scanY){
                  sum += featureMaps[centrIdx].getActivations(poolX * scanWidth + scanX, poolY * scanHeight + scanY);
               }
            }
            poolMaps[centrIdx].setPoolSum(poolX, poolY, sum);
         }
      }
   }
}

void ConvLayer::parseImagePatches(vector<Patch> patches){
   
}