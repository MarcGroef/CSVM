#ifndef CSVM_SETTINGS_H
#define CSVM_SETTINGS_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <csvm/csvm_svm.h>

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
      
      
  
   public:
      
      void readSettingsFile(string dir);
   };

}

#endif