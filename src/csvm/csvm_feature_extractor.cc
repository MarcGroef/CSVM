#include <csvm/csvm_feature_extractor.h>

using namespace std;
using namespace csvm;

//General feature extractor class, with delegates the feature extraction to the correct feature extractor (e.g. HoG)

FeatureExtractor::FeatureExtractor(){

    //settings.featureType = HOG;
}

Feature FeatureExtractor::extract(Patch p, unsigned int cbIdx){
	//cout << "extracting something" << endl;
   //settings.featureType = CLEAN;
   if(cbIdx < nLBP)
      return lbp[cbIdx].getLBP(p);
   cbIdx -= nLBP;
   if(cbIdx < nHOG)
      return hog[cbIdx].getHOG(p);
   cbIdx -= nHOG;
   if(cbIdx < nClean)
      return clean[cbIdx].describe(p);
   
   
   cout << "nCodebooks is higher than the amounst of feature extractors specified...\nExitting..\n";
   exit(-1);
   /*
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
   */
}

void FeatureExtractor::setSettings(FeatureExtractorSettings s){
   settings = s;

   
   nClean = s.clSettings.size();
   nHOG = s.hogSettings.size();
   nLBP = s.lbpSettings.size();
   
   lbp.resize(nLBP);
   hog.resize(nHOG);
   clean.resize(nClean);
   
   for(size_t clIdx = 0; clIdx != nClean; ++clIdx)
      clean[clIdx].settings = settings.clSettings[clIdx];
   
   for(size_t hIdx = 0; hIdx != nHOG; ++hIdx)
      hog[hIdx].setSettings(settings.hogSettings[hIdx]);
   
   for(size_t lbpIdx = 0; lbpIdx != nLBP; ++lbpIdx)
      lbp[lbpIdx].setSettings(settings.lbpSettings[lbpIdx]);


	   
}