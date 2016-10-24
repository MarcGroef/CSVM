#ifndef CSVM_CENTROID_H
#define CSVM_CENTROID_H

#include "csvm_feature.h"

//DEPRECATED

using namespace std;

namespace csvm{

   class Centroid{
      
   public:
      bool debugOut, normalOut;
     vector<float> content;
     float getDistanceSq(Feature f);
	 float getDistanceSq(Centroid c);
   };
   
   
}

#endif