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
      CSVMDatasetType type;
      unsigned int nImages;
   };
   
   class CSVMDataset{
      CSVMDataset_Settings settings;
      CIFAR10 cifar10;
      vector<Image> images;
      vector<int> testImagesIdx;
      vector< vector<unsigned int> > trainImagesIdx;   //[labelId][image]
      int nClasses;
   public:
      CSVMDataset();
      void loadCifar10(string labelsDir,vector<string> imageDirs);
      Image getImage(int index);
      Image* getImagePtr(int index);
      int getSize();
      void splitDatasetToClasses();
      int getNumberImagesInClass(int labelId);
      int getNumberClasses();
      void setSettings(CSVMDataset_Settings s);
      string getLabel(int labelId);
   };
}
#endif