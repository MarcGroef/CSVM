#ifndef CSVM_CLUSTER_ANALYSER_H
#define CSVM_CLUSTER_ANALYSER_H

#include <vector>

#include "csvm_feature.h"
#include "csvm_rbm.h"
#include "csvm_frequency_matrix.h"
#include "csvm_image.h"

using namespace std;
namespace csvm{
   
   enum CSVMClusterAnalyseMethods{
      CSVM_RBM,
      
   };
   
   
   struct ClusterAnalyserSettings{
      CSVMClusterAnalyseMethods method;
      RBMSettings rbmSettings;
      
   };
  
   class ClusterAnalyser{
      RBM rbm;
      FreqMatrix cooc;
      vector<Feature> images;
      ClusterAnalyserSettings settings;
   public:
      ClusterAnalyser();
      
      ~ClusterAnalyser();
      
      void setSettings(ClusterAnalyserSettings sets);
      void studyFeaturesRBM(vector<Feature> features);
      void studyCoocurences(vector<Feature> patchBatch);
   };

}
#endif