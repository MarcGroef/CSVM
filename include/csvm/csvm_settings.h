#ifndef CSVM_SETTINGS_H
#define CSVM_SETTINGS_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "csvm_svm.h"
#include "csvm_feature_extractor.h"
#include "csvm_codebook.h"
#include "csvm_image_scanner.h"
#include "csvm_dataset.h"

using namespace std;

namespace csvm{

   enum FEATURE_TYPE{
      CSVM_FEATURE_HOG = 1,
      
   };

   //this class should be able to read a settingsfile, or write a default settings file.
   //The settingsfile should contain all experiment parameters and relative directories of the datasets.

   class CSVMSettings{
      FEATURE_TYPE feature;
      SVM_Settings svmSettings;
      FeatureExtractorSettings featureSettings;
      Codebook_settings codebookSettings;
      ImageScannerSettings scannerSettings;
      CSVMDataset_Settings datasetSettings;
      
   public:
      void parseDatasetSettings(ifstream& stream);
      void readSettingsFile(string dir);
   };

}

#endif