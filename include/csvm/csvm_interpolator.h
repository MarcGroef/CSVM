#ifndef CSVM_INTERPOLATOR_H
#define CSVM_INTERPOLATOR_H

#include "csvm_image.h"
#include <vector>

using namespace std;

namespace csvm{
   
   class Interpolator{
      
   public:
      bool debugOut, normalOut;
      Image interpolate_bicubic(Image& im, unsigned int newWidth, unsigned int newHeight);
      
   };
   
}

#endif