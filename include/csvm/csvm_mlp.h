#ifndef CSVM_MLP_H
#define CSVM_MLP_H

#include <vector>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
   
 struct MLPSettings{
    //add your settings variables here (stuff you want to set through the settingsfile)
    unsigned int nHiddenUnits;
<<<<<<< HEAD
    unsigned int nInputUnits;
    unsigned int nOutputUnits;
 }
=======
 };
>>>>>>> 54887243eb652e62c8b74cf5effeea57f7e9f692
   
   
   
 class MLP{
    //class variables
    
    
 public:
    void train(vector<Feature>& randomFeatures);
    
    vector<double> getActivations(vector<Feature>& imageFeatures);
    
 };
   
}


#endif
