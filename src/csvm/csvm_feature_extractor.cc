#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

//General feature extractor class, with delegates the feature extraction to the correct feature extractor (e.g. HoG)

FeatureExtractor::FeatureExtractor(){

    //settings.featureType = HOG;
}

Feature FeatureExtractor::extract(Patch p){
	//cout << "extracting something" << endl;
   //settings.featureType = CLEAN;
   switch(settings.featureType){
      case LBP:
         return lbp.getLBP(p);
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
   clean.settings = settings.clSettings;
   if(settings.featureType == HOG)
      hog.setSettings(settings.hogSettings);
   if (settings.featureType == LBP)
	   lbp.setSettings(settings.lbpSettings);
   if (settings.featureType == MERGE) {
	   pixhog.setSettings(settings.mergeSettings);
	   hog.setSettings(settings.hogSettings);
   }
	   
}