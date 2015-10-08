#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H

#include "csvm_clean_descriptor.h"
#include "csvm_lbp_descriptor.h"
#include "csvm_feature.h"


using namespace std;
namespace csvm{
  
  enum FeatureType{
    LBP,
    CLEAN,
  };
  
  struct FeatureExtractorSettings{
    FeatureType featureType;
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    LBPDescriptor lbp;
    CleanDescriptor clean;
    
  public:
     FeatureExtractor();
     Feature extract(Patch p);
  };
}
#endif