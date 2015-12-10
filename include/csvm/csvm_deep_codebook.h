#ifndef CSVM_DEEP_CODEBOOK_H
#define CSVM_DEEP_CODEBOOK_H

#include <vector>
#include <cmath>

#include "csvm_feature.h"
#include "csvm_patch.h"
#include "csvm_centroid.h"

#include "csvm_kmeans.h"
#include "csvm_image_scanner.h"
#include "csvm_image.h"
#include "csvm_dataset.h"

using namespace std;

namespace csvm{
   
  
   
   class DeepCodebook{
      unsigned int nTotalImages;
      unsigned int nTrainImages;
      
      unsigned int nLayers;
      vector<unsigned int> fmSizes;
      vector<unsigned int> plSizes;
      vector<unsigned int> nCentroids;
      
      vector< vector< Centroid > > layerStack;
      KMeans kmeans;
      
      void calculateSizes(unsigned int imSize, unsigned int patchSize, unsigned int stride);
      ImageScanner* scanner;
      CSVMDataset* dataset;
      
   public:
      DeepCodebook(ImageScanner* imScanner, CSVMDataset* ds, unsigned int imSize, unsigned int patchSize, unsigned int stride);
      
   };
   
   
}
#endif