#ifndef CSVM_LVQ_H
#define CSVM_LVQ_H

//DEPRECATED

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include "csvm_feature.h"
#include <cmath>
#include <limits>

using namespace std;

namespace csvm{
  
    struct LVQ_Settings{
      int nClusters;
      float alpha;
    };

    class LVQ{
      LVQ_Settings settings;
      vector<Feature> initPrototypes(vector<Feature> collection, unsigned int labelId, unsigned int nProtos);
      
    public:
      bool debugOut, normalOut;
      LVQ();
      vector<Feature> cluster(vector<Feature> collection, unsigned int labelId, unsigned int numberPrototypes, float learningRate, int epochs);
    };
}

#endif
