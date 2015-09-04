
#ifndef HB_IMAGE_SCANNER_H
#define HB_IMAGE_SCANNER_H

#include <iostream>
#include <vector>

#include "hb_opencv_incl.h"

#include "hb_hog_descriptor.h"

using namespace cv;
using namespace std;

namespace hogbovw{



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
