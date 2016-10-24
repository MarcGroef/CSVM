#include <csvm/csvm_centroid.h>


//DEPRECATED
using namespace std;
using namespace csvm;

float Centroid::getDistanceSq(Feature f){
   unsigned int nDims = f.content.size();
   float sum = 0.0;
   for(size_t dIdx = 0; dIdx < nDims; ++dIdx){
      sum += (content[dIdx] - f.content[dIdx]) * (content[dIdx] - f.content[dIdx]);
   }
   return sum;
}


float Centroid::getDistanceSq(Centroid c) {
	unsigned int nDims = c.content.size();
	float sum = 0.0;
	for (size_t dIdx = 0; dIdx < nDims; ++dIdx) {
		sum += (content[dIdx] - c.content[dIdx]) * (content[dIdx] - c.content[dIdx]);
	}
	return sum;
}
