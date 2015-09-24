#include <csvm/csvm_dataset.h>

using namespace std;
using namespace csvm;


void CSVMDataset::loadCifar10(string labelsDir,vector<string> imageDirs){
   cifar10.readLabels(labelsDir);
   int imDirs = imageDirs.size();

   for(int i = 0; i < imDirs; i++){
      
      cifar10.loadImages(imageDirs[i]);
      
   }
   
}

Image CSVMDataset::getImage(int index){
   return cifar10.getImage(index);
}

Image* CSVMDataset::getImagePtr(int index){
   return cifar10.getImagePtr(index);
}

int CSVMDataset::getSize(){
   return cifar10.getSize();
}