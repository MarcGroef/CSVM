#include <csvm/csvm_dataset.h>

using namespace std;
using namespace csvm;

/* This class handles general dataset functionality. 
 * Given a image-index, this class is able to resolve the image and its label.
 * 
 * 
 * 
 * 
 */

CSVMDataset::CSVMDataset(){
   
}

CSVMDatasetType CSVMDataset::getType(){
   return settings.type;
}


void CSVMDataset::loadCifar10(string labelsDir,vector<string> imageDirs){
   cifar10.readLabels(labelsDir);
   int imDirs = imageDirs.size();

   for(int i = 0; i < imDirs; i++){
      
      cifar10.loadImages(imageDirs[i]);
      
   }
   //cout << "Loading cifar10 data. Splitting to classes..\n";
   //splitDatasetToClasses();
   //appendAndShuffleDataIdxArray();
   //cout << "dataset split to classes\n";
   if(settings.imWidth > 0 || settings.imHeight > 0)
      cifar10.scaleData(settings.imWidth, settings.imHeight);
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
   if(settings.imWidth > 0 || settings.imHeight > 0)
      mnist.scaleData(settings.imWidth,settings.imHeight);
   if(normalOut)
      cout << "Rescaled!\n";

}


//general call to load the dataset. Given the settings, it will load the correct one.
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
   splitDataset();
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

//get image based its real index
Image* CSVMDataset::getImagePtr(int index){
   switch(settings.type){
      case DATASET_CIFAR10:
         return cifar10.getImagePtr(index);
      case  DATASET_MNIST:
         return mnist.getImagePtr(index);
      
   }
   return cifar10.getImagePtr(index);
}

//get image based on the shuffled train-indices
Image* CSVMDataset::getTrainImagePtr(int trainIdx){
   return getImagePtr(trainImagesIdx[trainIdx]);
}

//get image based on the shuffled test-indices
Image* CSVMDataset::getTestImagePtr(int testIdx){
   return getImagePtr(testImagesIdx[testIdx]);
}


int CSVMDataset::getTrainSize(){
   
   return settings.nTrainImages;
}

int CSVMDataset::getTestSize(){
   return settings.nTestImages;
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

//shuffle image indices and slit for train/test
void CSVMDataset::splitDataset(){
   unsigned int nData = getTotalImages();
   vector<unsigned int> tempIdces(nData, 0);
   
   for(size_t imIdx = 0; imIdx < nData; ++imIdx)
      tempIdces[imIdx] = imIdx;
      
   for(size_t shuffleIdx = 0; shuffleIdx < nData; ++shuffleIdx){
      unsigned int randIdx = rand() % nData;
      unsigned int temp = tempIdces[shuffleIdx];
      tempIdces[shuffleIdx] = tempIdces[randIdx];
      tempIdces[randIdx] = temp;
   }
   
   unsigned int nTrainData = settings.nTrainImages;
   unsigned int nTestData = settings.nTestImages;
   
   if(nTrainData + nTestData > nData){
      cout << "csvm::CSVMDataset::splitDataset() WARNING! amount of testData + amount of trainData > nData (" << nTrainData << " + " << nTestData << " > " << nData << ")in dataset! Exitting..\n";
      exit(0);
   }
   
   
   trainImagesIdx.resize(nTrainData);
   testImagesIdx.resize(nTestData);

   
   for(size_t trainIdx = 0; trainIdx < nTrainData; ++trainIdx){
      trainImagesIdx[trainIdx] = tempIdces[trainIdx];
   }
   
   for(size_t testIdx = 0; testIdx < nTestData; ++testIdx){
      testImagesIdx[testIdx] = tempIdces[nTrainData + testIdx];
   }
         
   
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




