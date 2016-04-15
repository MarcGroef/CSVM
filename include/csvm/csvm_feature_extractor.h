#ifndef CSVM_FEATURE_EXTRACTOR_H
#define CSVM_FEATURE_EXTRACTOR_H

//General feature extractor class, with delegates the feature extraction to the correct feature extractor (e.g. HoG)

#include <cstdlib>

#include "csvm_clean_descriptor.h"
#include "csvm_lbp_descriptor.h"
#include "csvm_feature.h"
#include "csvm_hog_descriptor.h"
#include "csvm_merge_descriptor.h"
#include "csvm_lbp_descriptor.h"

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
    vector<HOGSettings> hogSettings;
    vector<CleanSettings> clSettings;
    vector<LBPSettings> lbpSettings;
  };
  
  class FeatureExtractor{
    FeatureExtractorSettings settings;
    vector<LBPDescriptor> lbp;
    vector<HOGDescriptor> hog;
    vector<CleanDescriptor> clean;
    vector<MERGEDescriptor> pixhog;
    
    unsigned int nLBP;
    unsigned int nHOG;
    unsigned int nClean;
    
    
  public:
     bool debugOut, normalOut;
     FeatureExtractor();
     Feature extract(Patch p, unsigned int cbIdx);
     void setSettings(FeatureExtractorSettings s);
  };
}
#endif