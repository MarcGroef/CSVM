#ifndef CSVM_DCB_FEATUREMAP_H
#define CSVM_DCB_FEATUREMAP_H

#include "../csvm_feature.h"


using namespace std;
namespace csvm{
   
   class FeatureMap{
      unsigned int width;
      unsigned int height;
      
      vector< vector< double> > fmap;
      
   public:
      FeatureMap(unsigned int width, unsigned int height);
      void setActivation(unsigned int pX, unsigned int pY, double value);
      double getActivations(unsigned int pX, unsigned int pY);
   };
   
}

#endif