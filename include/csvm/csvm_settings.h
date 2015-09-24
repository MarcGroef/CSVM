#ifndef CSVM_SETTINGS_H
#define CSVM_SETTINGS_H

#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;

namespace csvm{

   enum FEATURE_TYPE{
      CSVM_FEATURE_HOG = 1,
      
   };

   //this class should be able to read a settingsfile, or write a default settings file.
   //The settingsfile should contain all experiment parameters and relative directories of the datasets.

   class CSVMSettings{
      FEATURE_TYPE feature;
      struct SVM_settings;
      
   public:
      void readSettingsFile(string dir);
   };

}

#endif