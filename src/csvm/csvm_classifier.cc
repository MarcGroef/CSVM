#include <csvm/csvm_classifier.h>

using namespace std;
using namespace csvm;

CSVMClassifier::CSVMClassifier(){
   srand(time(NULL));
}

void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
}

void CSVMClassifier::constructCodebook(){
   int nPatches = 10000;
   pretrainDump.clear();
   pretrainDump.reserve(nPatches);
   
   vector<Patch> patches;
   vector<Feature> features;
   
   features.reserve(10);
   
  
   int nImages = dataset.getSize();

   for(int im = 0; im < 1000; ++im){
      patches = imageScanner.getRandomPatches(dataset.getImagePtr(rand() % nImages), 10, 8, 8);
      
      features.clear();
      
      for(size_t patchIdx = 0; patchIdx < 10; ++patchIdx)
         features.push_back(featExtr.extract(patches[patchIdx]));
      
      pretrainDump.insert(pretrainDump.end(),features.begin(),features.end());
      
   }
   cout << nPatches << "features extracted\n";
   
   codebook.constructCodebook(pretrainDump);
   
}