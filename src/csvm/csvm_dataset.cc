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
   //cout << "Loading cifar10 data. Splitting to classes..\n";
   splitDatasetToClasses();
   //appendAndShuffleDataIdxArray();
   //cout << "dataset split to classes\n";
}

void CSVMDataset::loadMNIST(string mnistDir){
   //cout << "loading mnist..\n";
   mnist.readTrainImages(mnistDir);
   mnist.readTrainLabels(mnistDir);
   mnist.readTestImages(mnistDir);
   mnist.readTestLabels(mnistDir);
   //cout << "read mnist data. Converting..\n";

   

   mnist.convertTrainSetToImages();
   //cout << "convert testset\n";
   mnist.convertTestSetToImages();
   
   //splitDatasetToClasses();
   //cout << "dataset split to classes\n";
   
}

void CSVMDataset::loadDataset(string dataDir){
   
   vector<string> imDirs;
   
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_1.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_2.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_3.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_4.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_5.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/test_batch.bin");
   
   
   
   switch(settings.type){
      case DATASET_CIFAR10:
         //cout << "Loading cifar10\n";
         loadCifar10(dataDir + "cifar-10-batches-bin/batches.meta.txt",imDirs);
         break;
      case  DATASET_MNIST:
         //cout << "laoding mnist\n";
         loadMNIST(dataDir + "mnist/");
         break;
      
   }
}

Image CSVMDataset::getImage(int index){
   switch(settings.type){
      case DATASET_CIFAR10:
         return cifar10.getImage(index);
      case  DATASET_MNIST:
         return mnist.getImage(index);
      
   }
   return cifar10.getImage(index);
}

Image* CSVMDataset::getImagePtr(int index){
   switch(settings.type){
      case DATASET_CIFAR10:
         return cifar10.getImagePtr(index);
      case  DATASET_MNIST:
         return mnist.getImagePtr(index);
      
   }
   return cifar10.getImagePtr(index);
}

int CSVMDataset::getSize(){
   
   return settings.nImages;
}

unsigned int CSVMDataset::getTotalImages(){
   switch(settings.type){
      case DATASET_CIFAR10:
         return cifar10.getSize();
      case  DATASET_MNIST:
         return mnist.getSize();
      
   }
   return cifar10.getSize();
}

void CSVMDataset::splitDatasetToClasses(){
   
   trainImagesIdx.clear();
   trainImagesIdx.resize(settings.nClasses);
   //unsigned int datasetSize = (unsigned int)cifar10.getSize();
   unsigned int datasetSize = 1000;
   int id;
   unsigned int image;
   
   for(size_t idx = 0; /*idx < settings.nImages  10000&&*/ idx < datasetSize; ++idx){
      if(settings.type == DATASET_CIFAR10)
         image = rand() % cifar10.getSize();
      else if (settings.type == DATASET_MNIST)
         image = rand() % mnist.getSize();
      
      id = (getImagePtr(image))->getLabelId();
      
      //cout << "ID = " << id << endl;
      trainImagesIdx[id].push_back(image);    
   }
   
}

void CSVMDataset::appendAndShuffleDataIdxArray(){

   unsigned int nData = 0;
   unsigned int rIdx;
   unsigned int buffer;
   unsigned int nWantedData = 1000;
   
   for(size_t clIdx = 0; clIdx < settings.nClasses; ++clIdx){
      nData += trainImagesIdx[clIdx].size();
   }
   
   vector<unsigned int> trainIndices(nData);
   
   for(size_t clIdx = 0; clIdx < settings.nClasses; ++clIdx){
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
   switch(settings.type){
      case DATASET_CIFAR10:
         return cifar10.getLabel(labelId);
      case  DATASET_MNIST:
         return mnist.getLabel(labelId);
      
   }
   return cifar10.getLabel(labelId);
}

int CSVMDataset::getNumberClasses(){
   return settings.nClasses;
}

Image* CSVMDataset::getImagePtrFromClass(unsigned int index, unsigned int classId){
   
   return getImagePtr(trainImagesIdx[classId][index]);
}

int CSVMDataset::getNumberImagesInClass(int labelId){
   return trainImagesIdx[labelId].size();
}
