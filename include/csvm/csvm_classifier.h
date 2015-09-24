#ifndef CSVM_CLASSIFIER_H
#define CSVM_CLASSIFIER_H

#include "csvm_settings.h"
#include "csvm_dataset.h"




using namespace std;
namespace csvm{
   
   class CSVMClassifier{
      CSVMSettings settings;
      
   public:
      //public vars
      CSVMDataset dataset;
      void setSettings(string settingsFile);
      
      //CSVMClassifier();
      
      
   };
   
}
#endif