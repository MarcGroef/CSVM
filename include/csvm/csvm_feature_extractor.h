#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H


#include "csvm_lbp_descriptor.h"
#include "csvm_feature.h"

using namespace std;
namespace csvm{
  
  enum FeatureType{
    LBP,
   
  };
  
  struct FeatureExtractorSettings{
    FeatureType featureType;
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    LBPDescriptor lbp;
    
  public:
     FeatureExtractor();
     Feature extract(Patch p);
  };
}
#endif