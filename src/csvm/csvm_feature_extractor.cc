#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

FeatureExtractor::FeatureExtractor(){

    //settings.featureType = HOG;
}

Feature FeatureExtractor::extract(Patch p){
	//cout << "extracting something" << endl;
   //settings.featureType = CLEAN;
   switch(settings.featureType){
      case LBP:
         return lbp.getLBP(p,0);
      case CLEAN:
         return clean.describe(p);
	  case HOG:
		  return hog.getHOG(p);
	  case MERGE:
		  return pixhog.getMERGE(p, clean , hog);
   }
   cout << "no featuretype set, retrieving HOG" << endl;
   return hog.getHOG(p);
}

void FeatureExtractor::setSettings(FeatureExtractorSettings s){
   settings = s;
   if(settings.featureType == HOG)
      hog.setSettings(settings.hogSettings);
   if (settings.featureType == MERGE) {
	   pixhog.setSettings(settings.mergeSettings);
	   hog.setSettings(settings.hogSettings);
   }
	   
}