#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H


using namespace std;
namespace csvm{
  
  enum FeatureType{
    HOG,
  }
  
  struct FeatureExtractorSettings{
    FeatureType featureType;
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    
    
  };
}
#endif