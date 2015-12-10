#ifndef CSVM_DATASET_H
#define CSVM_DATASET_H

#include <vector>
#include <cstdlib>
#include <ctime>

#include "csvm_cifar10_parser.h"
#include "csvm_image.h"
#include "csvm_mnist_parser.h"


using namespace std;
namespace csvm{
   
   enum CSVMDatasetType{
      DATASET_CIFAR10,
      DATASET_MNIST,
   };
   
   struct CSVMDataset_Settings{
      CSVMDatasetType type;
      unsigned int nImages;
      bool useDifferentCodebooksPerClass;
   };
   
   class CSVMDataset{
      CSVMDataset_Settings settings;
      CIFAR10 cifar10;
      MNISTParser mnistParser;
      vector<Image> images;
      vector<int> testImagesIdx;
      vector< vector<unsigned int> > trainImagesIdx;   //[labelId][image]
      vector<unsigned int> finalTrainIndices;
      int nClasses;
   public:
      CSVMDataset();
      void loadCifar10(string labelsDir,vector<string> imageDirs);
      void loadMNIST(string mnistDir);
      Image getImage(int index);
      Image* getImagePtr(int index);
      Image* getImagePtrFromClass(unsigned int index, unsigned int classId);
      int getSize();
      unsigned int getTotalImages();
      void splitDatasetToClasses();
      int getNumberImagesInClass(int labelId);
      int getNumberClasses();
      void setSettings(CSVMDataset_Settings s);
      string getLabel(int labelId);
      void appendAndShuffleDataIdxArray();
   };
}
#endif