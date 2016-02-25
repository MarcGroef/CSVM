#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <vector>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
   
 struct MLPSettings{
    //add your settings variables here (stuff you want to set through the settingsfile)
    unsigned int nHiddenUnits;
    unsigned int nInputUnits;
    unsigned int nOutputUnits;
 };

   
   
   
 class MLP{
 private:
    //class variables
    MLPSettings settings;
    
 public:
    void train(vector<Feature>& randomFeatures);
    
    vector<double> getActivations(vector<Feature>& imageFeatures);
    void setSettings(MLPSettings& s);
 };
   
}


#endif
