#include <csvm/deep_codebook/csvm_dcb_featuremap.h>

using namespace std;
using namespace csvm;


FeatureMap::FeatureMap(unsigned int width, unsigned int height){
   this->width = width;
   this->height = height;
   fmap = vector< vector<double> >(width, vector<double>(height));
}

void FeatureMap::setActivation(unsigned int pX, unsigned int pY, double value){
   fmap[pX][pY] = value;
}

double FeatureMap::getActivations(unsigned int pX, unsigned int pY){
   return fmap[pX][pY];
}