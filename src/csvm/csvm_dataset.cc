#include <csvm/csvm_dataset.h>

using namespace std;
using namespace csvm;


CSVMDataset::CSVMDataset(){

}

void CSVMDataset::loadCifar10(string labelsDir,vector<string> imageDirs){
   cifar10.readLabels(labelsDir);
   int imDirs = imageDirs.size();

   for(int i = 0; i < imDirs; i++){
      
      cifar10.loadImages(imageDirs[i]);
      
   }
   cout << "Loading cifar10 data. Splitting to classes..\n";
   splitDatasetToClasses();
   //appendAndShuffleDataIdxArray();
   //cout << "dataset split to classes\n";
}

void CSVMDataset::loadMNIST(string mnistDir){
   
   mnistParser.readTrainImages(mnistDir);
   mnistParser.readTrainLabels(mnistDir);
   mnistParser.readTestImages(mnistDir);
   mnistParser.readTestLabels(mnistDir);
   /*cifar10.readLabels(labelsDir);
   int imDirs = imageDirs.size();

   for(int i = 0; i < imDirs; i++){
      
      cifar10.loadImages(imageDirs[i]);
      
   }
   splitDatasetToClasses();
   //cout << "dataset split to classes\n";
   */
}

Image CSVMDataset::getImage(int index){
   return cifar10.getImage(index);
}

Image* CSVMDataset::getImagePtr(int index){
   return cifar10.getImagePtr(index);
}

int CSVMDataset::getSize(){
   return settings.nImages;
}

void CSVMDataset::splitDatasetToClasses(){
   nClasses = 10;
   trainImagesIdx.clear();
   trainImagesIdx.resize(nClasses);
   //unsigned int datasetSize = (unsigned int)cifar10.getSize();
   unsigned int datasetSize = 1000;
   int id;
   unsigned int image;
   
   for(size_t idx = 0; /*idx < settings.nImages  10000&&*/ idx < datasetSize; ++idx){
      image = rand() % cifar10.getSize();
      id = (cifar10.getImagePtr(image))->getLabelId();
      
      //cout << "ID = " << id << endl;
      trainImagesIdx[id].push_back(image);    
   }
   
}

void CSVMDataset::appendAndShuffleDataIdxArray(){
   unsigned int nClasses = 3;
   unsigned int nData = 0;
   unsigned int rIdx;
   unsigned int buffer;
   unsigned int nWantedData = 1000;
   
   for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
      nData += trainImagesIdx[clIdx].size();
   }
   
   vector<unsigned int> trainIndices(nData);
   
   for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
      trainIndices.insert(trainIndices.end(), trainImagesIdx[clIdx].begin(), trainImagesIdx[clIdx].end());
   }
   
   for(size_t dIdx = 0; dIdx < nData; ++dIdx){
      rIdx = rand() % nData;
      buffer = trainIndices[rIdx];
      trainIndices[rIdx] = trainIndices[dIdx];
      trainIndices[dIdx] = buffer;
   }
   
   finalTrainIndices.reserve(nWantedData);
   for(size_t wantedIdx = 0; wantedIdx < nWantedData; ++ wantedIdx)
      finalTrainIndices.push_back(trainIndices[wantedIdx]);
   
}

void CSVMDataset::setSettings(CSVMDataset_Settings s){
   settings = s;
}

string CSVMDataset::getLabel(int labelId){
   return cifar10.getLabel(labelId);
}

int CSVMDataset::getNumberClasses(){
   return nClasses;
}

Image* CSVMDataset::getImagePtrFromClass(unsigned int index, unsigned int classId){
   return cifar10.getImagePtr(trainImagesIdx[classId][index]);
}

int CSVMDataset::getNumberImagesInClass(int labelId){
   return trainImagesIdx[labelId].size();
}
