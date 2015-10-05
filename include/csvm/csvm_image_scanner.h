
#ifndef CSVM_IMAGE_SCANNER_H
#define CSVM_IMAGE_SCANNER_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "csvm_image.h"
#include "csvm_patch.h"



using namespace std;

namespace csvm{

   struct ImageScannerSettings{
      int stride;
      int patchWidth;
      int patchHeight;
      
   };


   class ImageScanner{
      ImageScannerSettings settings;
      
   public:
      ImageScanner();
      vector<Patch> scanImage(Image* image,unsigned int patchWidth,unsigned int patchHeight,unsigned int xStride,unsigned int yStride);
      vector<Patch> getRandomPatches(Image* image, unsigned int nPatches,unsigned int patchWidth, unsigned int patchHeight);
   };

}

#endif
