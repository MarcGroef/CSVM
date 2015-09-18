#ifndef CSVM_CIFAR10_PARSER_H
#define CSVM_CIFAR10_PARSER_H

#include <vector>
#include <iostream>

#include "csvm_image.h"

using namespace std;

namespace csvm{
   
   class CIFAR10{
      vector<Image> images;
      
      
   public:
      void loadImages(string dir);
   };
   
   
}

#endif