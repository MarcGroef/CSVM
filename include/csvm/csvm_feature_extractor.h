#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H

#include "csvm_lvq.h"
#include "csvm_kmeans.h"

using namespace std;
namespace csvm{
  
  enum FeatureType{
    HOG,
  };
  
  struct FeatureExtractorSettings{
    FeatureType featureType;
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    
    
  };
}
#endif