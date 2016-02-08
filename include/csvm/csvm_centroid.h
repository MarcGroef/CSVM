#ifndef CSVM_CENTROID_H
#define CSVM_CENTROID_H

#include "csvm_feature.h"

//DEPRECATED

using namespace std;

namespace csvm{

   class Centroid{
      
   public:
      bool debugOut, normalOut;
     vector<double> content;
     double getDistanceSq(Feature f);
	 double getDistanceSq(Centroid c);
   };
   
   
}

#endif