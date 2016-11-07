
#ifndef CSVM_IMAGE_SCANNER_H
#define CSVM_IMAGE_SCANNER_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "csvm_image.h"
#include "csvm_patch.h"


/* The image-scanner is responsible for extracting patches from images.
 * It can extract patches in a convolutional manner, using the patch-specifications in the settings-file.
 * 
 * It is also able to extract one patch from a given image, at a particular location.
 * 
 * An last, but not least, extract a patch at a random location of an image.
 * 
 * 
 * 
 */

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
      
      
   public:
      bool debugOut, normalOut;
      ImageScannerSettings settings;

      ImageScanner();
      void setSettings(ImageScannerSettings set);
      vector<Patch> scanImage(Image* image);
      Patch getRandomPatch(Image* image);
      Patch getPatchAt(Image* image, unsigned int x, unsigned int y);
	  void setScannerStride(unsigned int stride);
     unsigned int getScannerStride();
   };

}

#endif
