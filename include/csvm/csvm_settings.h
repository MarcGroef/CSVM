#ifndef CSVM_SETTINGS_H
#define CSVM_SETTINGS_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <cstdlib>
#include "csvm_svm.h"
#include "csvm_feature_extractor.h"
#include "csvm_codebook.h"
#include "csvm_image_scanner.h"
#include "csvm_dataset.h"
//#include "csvm_cluster_analyser.h"
#include "csvm_linear_network.h"
using namespace std;

namespace csvm{

   enum FEATURE_TYPE{
      CSVM_FEATURE_HOG = 1,
      
   };

   //this class should be able to read a settingsfile, or write a default settings file.
   //The settingsfile should contain all experiment parameters and relative directories of the datasets.

   class CSVMSettings{
     public:
      FEATURE_TYPE feature;
      SVM_Settings svmSettings;
      FeatureExtractorSettings featureSettings;
      Codebook_settings codebookSettings;
      ImageScannerSettings scannerSettings;
      CSVMDataset_Settings datasetSettings;
      LinNetSettings netSettings;
      //ClusterAnalyserSettings analyserSettings;
   
      ~CSVMSettings();
      void parseDatasetSettings(ifstream& stream);
      void readSettingsFile(string dir);
      //void parseClusterAnalserData(ifstream& stream);
      void parseCodebookSettings(ifstream& stream);
      void parseFeatureExtractorSettings(ifstream& stream);
      void parseImageScannerSettings(ifstream& stream);
      void parseSVMSettings(ifstream& stream);
   };

}

#endif