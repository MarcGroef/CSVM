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
         return lbp.getLBP(p,0);
      case CLEAN:
         return clean.describe(p);
	  case HOG:
		  return hog.getHOG(p);
	  
   }
   cout << "no featuretype set, retrieving HOG" << endl;
   return hog.getHOG(p);
}

void FeatureExtractor::setSettings(FeatureExtractorSettings s){
   settings = s;
   clean.settings = settings.clSettings;
   
   hog.setSettings(settings.hogSettings);
   
   
   
	   
}

void FeatureExtractor::flipType(){
   settings.featureType = settings.featureType2;
}