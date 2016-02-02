#ifndef CSVM_CIFAR10_PARSER_H
#define CSVM_CIFAR10_PARSER_H

/* This class contains functionality to read the CIFAr-10 binary files, parse them to the image class, scale images,
 * and make them available through the Dataset class.
 * 
 * 
 * 
 * 
 */

#include <vector>
#include <iostream>
#include <fstream>

#include "csvm_image.h"
#include "csvm_interpolator.h"



using namespace std;


namespace csvm{
      
   
   enum CSVM_CIFAR10_CONTSTANTS{
      N_IMAGES_PER_BATCH = 10000,
      IMAGE_SIZE = 3072,
   };
   
   class CIFAR10{
      vector<Image> images;
      vector<string> labels;
      Interpolator interpolator;
      Image bytesToImage(unsigned char* c);
   public:
      void readLabels(string dir);
      void loadImages(string dir);
      Image getImage(int index);
      Image* getImagePtr(int index);
      int getSize();
      string getLabel(int labelId);
      void scaleData(unsigned int width, unsigned int height);
   };
   
   
}

#endif