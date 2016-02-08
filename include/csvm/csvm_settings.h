#ifndef CSVM_SETTINGS_H
#define CSVM_SETTINGS_H

/* This settingsfile class parses the settingsfile, and wraps the info up to pass them 
 * to the other classes in the program that require them.
 * 
 * The methods of this class will be called by "CSVMClassifier in csvm_classifier.cc"
 * This is where the actual settings will be passed to the other classes.
 * 
 * The settings are put in a 'struct', so there each time one block of memory passed.
 * These structs are defined in the header files of the class where the're needed.
 * 
 * e.g. csvm_conv_svm.h contains the struct for convSVMSettings;
 * 
 * 
 */

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
      bool debugOut, normalOut;
      CLASSIFIER classifier;
      CODEBOOK codebook;
      
      
      FEATURE_TYPE feature;
      SVM_Settings svmSettings;
      FeatureExtractorSettings featureSettings;
      Codebook_settings codebookSettings;
      ImageScannerSettings scannerSettings;
      CSVMDataset_Settings datasetSettings;
      LinNetSettings netSettings;
      ConvSVMSettings convSVMSettings;
      DCBSettings dcbSettings; //deep codebook
      
      //ClusterAnalyserSettings analyserSettings;
   
      ~CSVMSettings();
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