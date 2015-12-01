#include <csvm/csvm_centroid.h>

using namespace std;
using namespace csvm;

double Centroid::getDistanceSq(Feature f){
   unsigned int nDims = f.content.size();
   double sum = 0.0;
   for(size_t dIdx = 0; dIdx < nDims; ++dIdx){
      sum += (content[dIdx] - f.content[dIdx]) * (content[dIdx] - f.content[dIdx]);
   }
   return sum;
}