#ifndef CSVM_CENTROID_H
#define CSVM_CENTROID_H

#include "csvm_feature.h"

using namespace std;

namespace csvm{

   class Centroid{
      
   public:
     vector<double> content;
     double getDistanceSq(Feature f);
	 double getDistanceSq(Centroid c);
   };
   
   
}

#endif