#include <csvm/deep_codebook/csvm_dcb_poolmap.h>

using namespace std;
using namespace csvm;

PoolMap::PoolMap(unsigned int width, unsigned int height){
   this->width = width;
   this->height = height;
   pMap = vector< vector<double> >(width, vector<double>(height));
}

void PoolMap::setPoolSum(unsigned int pX, unsigned int pY, double value){
   pMap[pX][pY] = value;
}

double PoolMap::getPoolSum(unsigned int pX, unsigned int pY){
   return pMap[pX][pY];
}