
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
      Mat image;
      string imageDir;
      
      
      Mat window;
      int winSize;
      int winPosX;
      int winPosY;

      int nPatches;

      vector< vector<float> > v_hogValues;

      public:
      ImageScanner(int wSize,int nPatches);
      void setImage(string filename);
      void showImage();
      void scanImage();

   };

}

#endif
