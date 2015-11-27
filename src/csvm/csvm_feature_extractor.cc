#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

FeatureExtractor::FeatureExtractor(){

    //settings.featureType = HOG;
}

Feature FeatureExtractor::extract(Patch p){
   //settings.featureType = CLEAN;
   switch(settings.featureType){
      case LBP:
         return lbp.getLBP(p,0);
      case CLEAN:
         return clean.describe(p);
	  case HOG:
		  return hog.getHOG(p, 0, true);
   }
   
   return lbp.getLBP(p,0);
}

void FeatureExtractor::setSettings(FeatureExtractorSettings s){
   settings = s;
   if(settings.featureType == HOG)
      hog.setSettings(settings.hogSettings);
}