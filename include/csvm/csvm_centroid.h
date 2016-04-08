#ifndef CSVM_CENTROID_H
#define CSVM_CENTROID_H

#include <cmath>
#include "csvm_feature.h"
#include "csvm_interpolator.h"

using namespace std;

namespace csvm{

   class Centroid{
      
   public:
    bool debugOut, normalOut;
    vector<double> content;
    double getDistanceSq(Feature f);
	 double getDistanceSq(Centroid c);
    void exportToPNG(string name, unsigned int nChannels);
   };
   
   
}

#endif