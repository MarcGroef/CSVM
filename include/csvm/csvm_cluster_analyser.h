#ifndef CSVM_CLUSTER_ANALYSER_H
#define CSVM_CLUSTER_ANALYSER_H

#include <vector>

#include "csvm_feature.h"
#include "csvm_rbm.h"
#include "csvm_frequency_matrix.h"
#include "csvm_image.h"

using namespace std;
namespace csvm{
  
   class ClusterAnalyser{
      RBM rbm;
      vector<Feature> images;
      
   public:
      ClusterAnalyser();
      
      ~ClusterAnalyser();
      void studyFeatures(vector<Feature> features);
   };

}
#endif