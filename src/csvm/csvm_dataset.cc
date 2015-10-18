#include <csvm/csvm_dataset.h>

using namespace std;
using namespace csvm;


CSVMDataset::CSVMDataset(){
    settings.nImages = 10000;
}

void CSVMDataset::loadCifar10(string labelsDir,vector<string> imageDirs){
   cifar10.readLabels(labelsDir);
   int imDirs = imageDirs.size();

   for(int i = 0; i < imDirs; i++){
      
      cifar10.loadImages(imageDirs[i]);
      
   }
   splitDatasetToClasses();
   //cout << "dataset split to classes\n";
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
   unsigned int datasetSize = (unsigned int)cifar10.getSize();
   int id;
   
   
   for(size_t idx = 0; idx < settings.nImages && idx < datasetSize; ++idx){
      id = (cifar10.getImagePtr(idx))->getLabelId();
      trainImagesIdx[id].push_back(idx);    
   }
   
}

int CSVMDataset::getNumberClasses(){
   return nClasses;
}

int CSVMDataset::getNumberImagesInClass(int labelId){
   return trainImagesIdx[labelId].size();
}