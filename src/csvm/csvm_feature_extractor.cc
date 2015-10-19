#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

FeatureExtractor::FeatureExtractor(){
    settings.featureType = LBP;//CLEAN;  
}

Feature FeatureExtractor::extract(Patch p){
   
   switch(settings.featureType){
      case LBP:
         return lbp.getLBP(p,0);
      case CLEAN:
         return clean.describe(p);
   }
   
   return lbp.getLBP(p,0);
}

