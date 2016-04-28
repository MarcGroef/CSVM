#ifndef CSVM_DEEP_CODEBOOK_H
#define CSVM_DEEP_CODEBOOK_H


// This class contains the deep codebook.
// calcSimilarity calculates activations from a feature to a centroid
// calculateSizes sets the layer sizes and the amount of layers. (The general architecture info)

#include <vector>
#include <cmath>
#include <iostream>

#include "csvm_feature.h"
#include "csvm_patch.h"
#include "csvm_centroid.h"

#include "csvm_kmeans.h"
#include "csvm_image_scanner.h"
#include "csvm_image.h"
#include "csvm_dataset.h"
#include "csvm_feature_extractor.h"
#include <cstdlib>

using namespace std;

namespace csvm{
   
   enum ActFunction{
     DCB_RBF,
     DCB_SOFT_ASSIGNMENT,
     DCB_SOFT_ASSIGNMENT_CLIPPING,
	  DCB_COSINE_SOFT_ASSIGNMENT,
   };
   
   enum Architecture{
     DCB_ALPHA,
     DCB_BETA,
     DCB_GAMMA,
   };
   
   struct DCBSettings{
      ActFunction simFunction;
      Architecture architecture;
      double similaritySigma;
      unsigned int nCentroids;
      unsigned int nRandomPatches;
      unsigned int nIter;
   };
  
   
   class DeepCodebook{
      DCBSettings settings;
      
      unsigned int nTotalImages;
      unsigned int nTrainImages;
      
      unsigned int nLayers;
      vector<unsigned int> fmSizes;
      vector<unsigned int> plSizes;
      vector<unsigned int> nCentroids;
      vector<unsigned int> nRandomPatches;
      
      vector< vector< Centroid > > layerStack;
      KMeans kmeans;
      
      void calculateSizes(unsigned int imSize, unsigned int patchSize, unsigned int stride);
      ImageScanner* scanner;
      CSVMDataset* dataset;
      FeatureExtractor* featExtr;
      
      vector<double> calcSimilarity(Feature& p, vector<Centroid>& c);
      vector<double> calculatePoolMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y);
      vector<double> calculateConvMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y);
   public:
      bool debugOut, normalOut;
      DeepCodebook(FeatureExtractor* fe, ImageScanner* imScanner, CSVMDataset* ds);
      void setSettings(DCBSettings& s);
      void generateCentroids();
      vector<double> getActivations(Image* im);
   };
   
   
}
#endif