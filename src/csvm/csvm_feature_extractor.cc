#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

FeatureExtractor::FeatureExtractor(){
<<<<<<< HEAD
    settings.featureType = LBP;//CLEAN;  
=======
    settings.featureType = LBP;
>>>>>>> 8ae3a8dc116a3bf7d630508db36e0690bdd72e06
}

Feature FeatureExtractor::extract(Patch p){
   
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

