#ifndef CSVM_DATASET_H
#define CSVM_DATASET_H

/* This class handles general dataset functionality. 
 * Given a image-index, this class is able to resolve the image and its label.
 * 
 * 
 * 
 * 
 */

#include <vector>
#include <cstdlib>
#include <ctime>

#include "csvm_cifar10_parser.h"
#include "csvm_image.h"
#include "csvm_mnist_parser.h"
#include "csvm_interpolator.h"

using namespace std;
namespace csvm{
   
   enum CSVMDatasetType{
      DATASET_CIFAR10,
      DATASET_MNIST,
   };
   
   struct CSVMDataset_Settings{
      CSVMDatasetType type;
      unsigned int nTrainImages;
      unsigned int nTestImages;
      bool useDifferentCodebooksPerClass;
      unsigned int nClasses;
      unsigned int imWidth;
      unsigned int imHeight;
   };
   
   class CSVMDataset{
      CSVMDataset_Settings settings;
      CIFAR10 cifar10;
      MNISTParser mnist;
      
      vector<unsigned int> testImagesIdx;
      vector<unsigned int> trainImagesIdx;  
      vector<unsigned int> finalTrainIndices;

   public:
      bool debugOut, normalOut;
      CSVMDataset();
      CSVMDatasetType getType();
      void loadCifar10(string labelsDir,vector<string> imageDirs);
      void loadMNIST(string mnistDir);
      Image getImage(int index);
      Image* getImagePtr(int index);
      Image* getTrainImagePtr(int trainIdx);
      Image* getTestImagePtr(int testIdx);
      void setTrainImages(vector<unsigned int> listOfImageNums);
      void setTestImages(vector<unsigned int> listOfImageNums);
      vector<unsigned int> getTrainImageNums();
      vector<unsigned int> getTestImageNums();
      unsigned int getTrainImageIdx(int trainIdx);

      int getTrainSize();
      int getTestSize();
      unsigned int getTotalImages();
      int getNumberClasses();
      void setSettings(CSVMDataset_Settings s);
      string getLabel(int labelId);
      void loadDataset(string dataDir);
      void splitDataset();
   };
}
#endif
