#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H

#include "csvm_clean_descriptor.h"
#include "csvm_lbp_descriptor.h"
#include "csvm_feature.h"
#include "csvm_hog_descriptor.h"
#include "csvm_merge_descriptor.h"

using namespace std;
namespace csvm{
  
  enum FeatureType{
    LBP,
    CLEAN,
    HOG,
	MERGE,
  };
  
  struct FeatureExtractorSettings{
    FeatureType featureType;
    HOGSettings hogSettings;

	MERGESettings mergeSettings;	//
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    LBPDescriptor lbp;
	 HOGDescriptor hog;
    CleanDescriptor clean;
	MERGEDescriptor pixhog;
    
  public:
     FeatureExtractor();
     Feature extract(Patch p);
     void setSettings(FeatureExtractorSettings s);
  };
}
#endif