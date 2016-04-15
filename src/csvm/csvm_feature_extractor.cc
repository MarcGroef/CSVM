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
<<<<<<< HEAD
   clean.settings = settings.clSettings[0];
   if(settings.featureType[0] == HOG)
      hog.setSettings(settings.hogSettings[0]);
   if (settings.featureType[0] == LBP)
	   lbp.setSettings(settings.lbpSettings[0]);
=======
   
   nClean = s.clean.size();
   nHOG = s.hog.size();
   nLBP = s.lbp.size();
   
   lbp.resize(nLBP);
   hog.resize(nHOG);
   nClean.resize(nClean);
   
   for(size_t clIdx = 0; clIdx != nClean; ++clIdx)
      clean[clIdx].settings = settings.clSettings[clIdx];
   
   for(size_t hIdx = 0; hIdx != nHOG; ++hIdx)
      hog[hIdx].setSettings(settings.hogSettings[hIdx]);
   
   for(size_t lbpIdx = 0; lbpIdx != nLBP; ++lbpIdx)
      lbp[lbpIdx].setSettings(settings.lbpSettings[lbpIdx]);

>>>>>>> c31dbd797cc0277564b71b81b049ccdbe2944bf1
	   
}