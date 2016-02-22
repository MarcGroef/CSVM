#ifndef CSVM_WHITENING_H
#define CSVM_WHITENING_H

#include <vector>
#include "csvm_feature.h"

using namespace std;

namespace csvm{
   
   class Whitener{
      vector<double> means;
      vecor< vector<double> > sigma;
   };
   
   
}

#endif