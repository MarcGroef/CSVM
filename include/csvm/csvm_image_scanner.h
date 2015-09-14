
#ifndef CSVM_IMAGE_SCANNER_H
#define CSVM_IMAGE_SCANNER_H

#include <iostream>
#include <vector>

#include "csvm_image.h"

#include "csvm_hog_descriptor.h"


using namespace std;

namespace csvm{



   class ImageScanner{
      Image image;                                //openCV Image container to hold an image
      string imageDir;                          //directory of the image
      
      
      Image window;                               //the window sample
      int winSize;
      int winPosX;
      int winPosY;

      int nPatches;                             //number of patches that should be taken

      vector< vector<double> > v_hogValues;      // vector of hog vectors

      public:
      ImageScanner(int wSize, int nPatches);     //contructor
      void setImage(string filename);           //load image into ImageScanner
      void scanImage();                         //scan the image

   };

}

#endif
