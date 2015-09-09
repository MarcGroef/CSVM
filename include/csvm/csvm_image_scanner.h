
#ifndef CSVM_IMAGE_SCANNER_H
#define CSVM_IMAGE_SCANNER_H

#include <iostream>
#include <vector>

#include "csvm_opencv_incl.h"

#include "csvm_hog_descriptor.h"

using namespace cv;
using namespace std;

namespace csvm{



   class ImageScanner{
      Mat image;                                //openCV Image container to hold an image
      string imageDir;                          //directory of the image
      
      
      Mat window;                               //the window sample
      int winSize;
      int winPosX;
      int winPosY;

      int nPatches;                             //number of patches that should be taken

      vector< vector<float> > v_hogValues;      // vector of hog vectors

      public:
      ImageScanner(int wSize,int nPatches);     //contructor
      void setImage(string filename);           //load image into ImageScanner
      void showImage();                         //popup a window to show the image
      void scanImage();                         //scan the image

   };

}

#endif
