#ifndef CSVM_CLASSIFIER_H
#define CSVM_CLASSIFIER_H

#include "csvm_settings.h"
#include "csvm_dataset.h"
#include "csvm_codebook.h"
#include "csvm_image_scanner.h"



using namespace std;
namespace csvm{
   
   class CSVMClassifier{
      CSVMSettings settings;
      Codebook codebook;
      ImageScanner imageScanner;
      
      
   public:
      //public vars
      CSVMDataset dataset;
      void setSettings(string settingsFile);
      void constructCodebook()
      //CSVMClassifier();
      
      
   };
   
}
#endif