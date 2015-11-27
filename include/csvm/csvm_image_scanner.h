
#ifndef CSVM_IMAGE_SCANNER_H
#define CSVM_IMAGE_SCANNER_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "csvm_image.h"
#include "csvm_patch.h"

//TODO: Add zero padding while scanning.

using namespace std;

namespace csvm{

   struct ImageScannerSettings{
      unsigned int stride;
      unsigned int patchWidth;
      unsigned int patchHeight;
      unsigned int nRandomPatches;
      bool useDifferentCodebooksPerClass;
   };


   class ImageScanner{
      ImageScannerSettings settings;
      
   public:
      ImageScanner();
      void setSettings(ImageScannerSettings set);
      vector< vector<Patch> > scanImage(Image* image);
      vector<Patch> getRandomPatches(Image* image);
   };

}

#endif
