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
   switch(settings.featureType[0]){
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
   clean.settings = settings.clSettings[0];
   if(settings.featureType[0] == HOG)
      hog.setSettings(settings.hogSettings[0]);
   if (settings.featureType[0] == LBP)
	   lbp.setSettings(settings.lbpSettings[0]);
	   
}