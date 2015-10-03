#ifndef CSVM_LVQ_H
#define CSVM_LVQ_H

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
      
    public:
      vector<Feature> cluster(vector<Feature> collection);
    };
}

#endif
