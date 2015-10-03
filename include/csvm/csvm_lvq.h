#ifndef CSVM_LVQ_H
#define CSVM_LVQ_H
#include <ctime>
#include <cstdlib>
#include <vector>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
  
    struct LVQ_Settings{
      int nClusters;
      double alpha;
    };

    class LVQ{
      LVQ_Settings settings;
      vector<Feature> initPrototypes(vector<Feature> collection, unsigned int nProtos);
      
    public:
      LVQ();
      vector<Feature> cluster(vector<Feature> collection, unsigned int numberPrototypes, double learningRate);
    };
}

#endif
