#include <csvm/csvm_classifier.h>

using namespace std;
using namespace csvm;

CSVMClassifier::CSVMClassifier(){
   srand(time(NULL));
   
}

void CSVMClassifier::initSVMs(){
   double learningRate = 0.05;
   svms.reserve(codebook.getNClasses());
   for(size_t svmIdx = 0; svmIdx < codebook.getNClasses(); ++svmIdx){
      svms.push_back(SVM(dataset.getSize(), codebook.getNClasses() * codebook.getNCentroids(), learningRate, svmIdx, codebook.getNCentroids()));
   }
}


void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
   analyser.setSettings(settings.analyserSettings);
   imageScanner.setSettings(settings.scannerSettings);
}


void CSVMClassifier::exportCodebook(string filename){
   codebook.exportCodebook(filename);
}

void CSVMClassifier::importCodebook(string filename){
   codebook.importCodebook(filename);
}


void CSVMClassifier::constructCodebook(){
   unsigned int nPatches = 10;  //number of random patches from each image
   unsigned int nClasses = dataset.getNumberClasses();
   

   
   vector<Patch> patches;
   vector<Feature> features;
   
   features.reserve(nPatches);
   
   pretrainDump.clear();
   pretrainDump.resize(nClasses);
   unsigned int nImages = dataset.getSize();
   cout << "constructing codebooks for " << nClasses << " classes using " << nImages << " images in total\n";
   for(size_t cl = 0; cl < nClasses; ++cl){
      
      //cout << "parsing class " << cl << endl;
      nImages = dataset.getNumberImagesInClass(cl);
      //cout << nImages << " images\n";
      
      pretrainDump[cl].reserve(nPatches * nImages);
      //cout << "space reserved\n";
      for(size_t im = 0; im < nImages; ++im){
         //cout << "scanning patches\n";
         patches = imageScanner.getRandomPatches(dataset.getImagePtr(im), nPatches, 8, 8);
         
         features.clear();
         //cout << "extracting patches..\n";
         for(size_t patchIdx = 0; patchIdx < nPatches; ++patchIdx)
            features.push_back(featExtr.extract(patches[patchIdx]));
         
         pretrainDump[cl].insert(pretrainDump[cl].end(),features.begin(),features.end());
         
      }
      //cout << pretrainDump[cl].size() << " features extracted for class " << cl << "\n";
      codebook.constructCodebook(pretrainDump[cl],cl);
      cout << "done constructing codebook for class " << cl << " using " << nImages << " images, " << nPatches << " patches each: " << nPatches * nImages<<" in total.\n";
   }
   pretrainDump.clear();
}

void CSVMClassifier::trainSVMs(){
   unsigned int datasetSize = dataset.getSize();
   vector < vector < Feature > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx), 8, 8, 1, 1);
      
      //clear previous features
      dataFeatures.clear();
      //allocate for new
      dataFeatures.reserve(patches.size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
      patches.clear();
      
      Feature* f = &dataFeatures[0];
      
      //cout << "datacontent: " << f->size << endl;
      //get cluster activations for the features
      //cout << "pushing back " << codebook.getActivations(dataFeatures)[0].content.size() << "act feats\n";
      datasetActivations.push_back(codebook.getActivations(dataFeatures));
   }
   cout << "I can get activations me!\n";
   //train the SVMs with the gained activations
   for(size_t svmIdx = 0; svmIdx < svms.size(); ++svmIdx){
      svms[svmIdx].train(datasetActivations);
   }
}