#ifndef CSVM_CIFAR10_PARSER_H
#define CSVM_CIFAR10_PARSER_H

#include <vector>
#include <iostream>
#include <fstream>

#include "csvm_image.h"

using namespace std;


namespace csvm{
      
   
   enum CSVM_CIFAR10_CONTSTANTS{
      N_IMAGES_PER_BATCH = 10000,
      IMAGE_SIZE = 3073,
   };
   
   class CIFAR10{
      vector<Image> images;
      vector<string> labels;
      
      Image bytesToImage(unsigned char* c);
   public:
      void readLabels(string dir);
      void loadImages(string dir);
   };
   
   
}

#endif