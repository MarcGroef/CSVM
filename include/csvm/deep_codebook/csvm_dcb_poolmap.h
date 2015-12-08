#ifndef CSVM_DCB_POOLMAP_H
#define CSVM_DCB_POOLMAP_H

#include <vector>

using namespace std;

namespace csvm{
   class PoolMap{
      unsigned int width;
      unsigned int height;
      vector< vector<double> > pMap;
      
   public:
      PoolMap(unsigned int width, unsigned int height);
      void setPoolSum(unsigned int pX, unsigned int pY, double value);
      double getPoolSum(unsigned int pX, unsigned int pY);
   };
}

#endif