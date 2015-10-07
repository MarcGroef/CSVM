#include <csvm/csvm_classifier.h>

using namespace std;
using namespace csvm;

CSVMClassifier::CSVMClassifier(){
   srand(time(NULL));
}



void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
   analyser.setSettings(settings.analyserSettings);
   imageScanner.setSettings(settings.scannerSettings);
}

void CSVMClassifier::trainRBM(){
   
   unsigned int nImages = dataset.getSize();
   cout << "Collecting RBM data ...\n";
   vector<Patch> patches;
   vector<Feature> features;
   bool dumpAlloced = false;
   pretrainDump.clear();
   unsigned int nPatches;
   
   //nImages = 100;
   unsigned int batchSize = 1000;
   for(size_t batch = 0; batch+batchSize < nImages; batch+= batchSize){
      cout << "Processing batch " << batch/batchSize << endl;
      for(size_t idx = batch; idx < batch + batchSize; ++idx){
         //cout << "processing image " << idx << endl;
         patches = imageScanner.scanImage(dataset.getImagePtr(idx),8,8,1,1);
         features.clear();
         
         if(!dumpAlloced){
            nPatches = patches.size();
            //cout << "allocated " << nPatches << " patches\n";
            pretrainDump.reserve(batchSize * nPatches);
            features.reserve(nPatches);
            dumpAlloced = true;
         }
         
         for(size_t patchIdx = 0; patchIdx < nPatches; ++patchIdx){
            Feature feat = featExtr.extract(patches[patchIdx]);
            features.push_back(codebook.getActivations(&feat));
            
         }
         //cout << "inserting " << features.size() << " features to dump\n";
         pretrainDump.insert(pretrainDump.end(),features.begin(),features.end());
      }
      cout << "Training RBM batch " << batch/batchSize  <<" out of " << (nImages/batchSize) << " on " << pretrainDump.size() << " features ..\n";
      analyser.studyFeatures(pretrainDump);
      pretrainDump.clear();
      dumpAlloced=false;
   }
   features.clear();
}

void CSVMClassifier::constructCodebook(){
   int nPatches = 10;
   pretrainDump.clear();
   pretrainDump.reserve(nPatches * dataset.getSize());
   
   vector<Patch> patches;
   vector<Feature> features;
   
   features.reserve(nPatches);
   
  
   int nImages = dataset.getSize();
   //nImages = 10;
   for(int im = 0; im < nImages; ++im){
      patches = imageScanner.getRandomPatches(dataset.getImagePtr(rand() % nImages), nPatches, 8, 8);
      
      features.clear();
      
      for(size_t patchIdx = 0; patchIdx < nPatches; ++patchIdx)
         features.push_back(featExtr.extract(patches[patchIdx]));
      
      pretrainDump.insert(pretrainDump.end(),features.begin(),features.end());
      
   }
   cout << pretrainDump.size() << " features extracted\n";
   
   codebook.constructCodebook(pretrainDump);
   
}