#include <csvm/csvm_classifier.h>

using namespace std;
using namespace csvm;

CSVMClassifier::CSVMClassifier(){
   srand(time(NULL));
}

void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
}

void CSVMClassifier::trainRBM(){
   cout << "Collecting RBM data...\n";
   unsigned int nImages = dataset.getSize();
   vector<Patch> patches;
   vector<Feature> features;
   bool dumpAlloced = false;
   pretrainDump.clear();
   unsigned int nPatches;
   
   //nImages = 100;
   
   for(size_t idx = 0; idx < nImages; ++idx){
      patches = imageScanner.scanImage(dataset.getImagePtr(idx),8,8,1,1);
      features.clear();
      
      if(!dumpAlloced){
         nPatches = patches.size();
         pretrainDump.reserve(nImages * nPatches);
         features.reserve(nPatches);
         dumpAlloced = true;
      }
      
      for(size_t patchIdx = 0; patchIdx < nPatches; ++patchIdx){
         Feature feat = featExtr.extract(patches[patchIdx]);
         features.push_back(codebook.getActivations(&feat));
         
      }
      
      pretrainDump.insert(pretrainDump.end(),features.begin(),features.end());
   }
   cout << "Training RBM..\n";
   analyser.studyFeatures(pretrainDump);
   pretrainDump.clear();
   features.clear();
}

void CSVMClassifier::constructCodebook(){
   int nPatches = 1000;
   pretrainDump.clear();
   pretrainDump.reserve(nPatches);
   
   vector<Patch> patches;
   vector<Feature> features;
   
   features.reserve(10);
   
  
   int nImages = dataset.getSize();
   
   for(int im = 0; im < 100; ++im){
      patches = imageScanner.getRandomPatches(dataset.getImagePtr(rand() % nImages), 10, 8, 8);
      
      features.clear();
      
      for(size_t patchIdx = 0; patchIdx < 10; ++patchIdx)
         features.push_back(featExtr.extract(patches[patchIdx]));
      
      pretrainDump.insert(pretrainDump.end(),features.begin(),features.end());
      
   }
   cout << nPatches << " features extracted\n";
   
   codebook.constructCodebook(pretrainDump);
   
}