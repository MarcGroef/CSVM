#ifndef CSVM_DATASET_H
#define CSVM_DATASET_H

#include <vector>

#include "csvm_cifar10_parser.h"
#include "csvm_image.h"


using namespace std;
namespace csvm{
   
   enum CSVMDatasetType{
      DATASET_CIFAR10,
      
   };
   
   struct CSVMDataset_Settings{
      int nImages;
   };
   
   class CSVMDataset{
      CIFAR10 cifar10;
      vector<Image> images;
      
   public:
      void loadCifar10(string labelsDir,vector<string> imageDirs);
      Image getImage(int index);
      Image* getImagePtr(int index);
      int getSize();
   };
}
#endif