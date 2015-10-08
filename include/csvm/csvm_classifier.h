#ifndef CSVM_CLASSIFIER_H
#define CSVM_CLASSIFIER_H

#include <vector>
#include <cstdlib>
#include <ctime>

#include "csvm_settings.h"
#include "csvm_dataset.h"
#include "csvm_codebook.h"
#include "csvm_image_scanner.h"
#include "csvm_cluster_analyser.h"
#include "csvm_feature.h"
#include "csvm_feature_extractor.h"

using namespace std;
namespace csvm{
   
   class CSVMClassifier{
      CSVMSettings settings;
      Codebook codebook;
      ImageScanner imageScanner;
      ClusterAnalyser analyser;
      FeatureExtractor featExtr;
      vector<Feature> pretrainDump;
   public:
      //public vars
      CSVMDataset dataset;
      
      CSVMClassifier();
      void setSettings(string settingsFile);
      void constructCodebook();
      void exportCodebook(string filename);
      void importCodebook(string filename);
      void trainRBM();
      //CSVMClassifier();
      
      
   };
   
}
#endif