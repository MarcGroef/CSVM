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
#include "csvm_conv_svm.h"
//#include "csvm_cluster_analyser.h"
#include "csvm_linear_network.h"
#include "csvm_deep_codebook.h"
#include "csvm_new_rbm.h"

using namespace std;

namespace csvm{

   enum FEATURE_TYPE{
      CSVM_FEATURE_HOG = 1,
      
   };
   
   enum CLASSIFIER{
      CL_SVM,
      CL_CSVM,
      CL_LINNET,
   };
   
   enum CODEBOOK{
      CB_CODEBOOK,
      CB_DEEPCODEBOOK,
   };

   //this class should be able to read a settingsfile, or write a default settings file.
   //The settingsfile should contain all experiment parameters and relative directories of the datasets.

   class CSVMSettings{
     public:
      CLASSIFIER classifier;
      CODEBOOK codebook;
      bool debugOut, normalOut;
      bool useRBM;      
      FEATURE_TYPE feature;
      SVM_Settings svmSettings;
      FeatureExtractorSettings featureSettings;
      Codebook_settings codebookSettings;
      ImageScannerSettings scannerSettings;
      CSVMDataset_Settings datasetSettings;
      LinNetSettings netSettings;
      ConvSVMSettings convSVMSettings;
      DCBSettings dcbSettings; //deep codebook
      NRBMSettings rbmSettings;
      //ClusterAnalyserSettings analyserSettings;
   
      ~CSVMSettings();
      
      void parseRBMSettings(ifstream& stream);
      void parseDatasetSettings(ifstream& stream);
      void readSettingsFile(string dir);
      //void parseClusterAnalserData(ifstream& stream);
      void parseCodebookSettings(ifstream& stream);
      void parseFeatureExtractorSettings(ifstream& stream);
      void parseImageScannerSettings(ifstream& stream);
      void parseSVMSettings(ifstream& stream);
      void parseLinNetSettings(ifstream& stream);
      void parseConvSVMSettings(ifstream& stream);
      void parseGeneralSettings(ifstream& stream);
   };

}

#endif
