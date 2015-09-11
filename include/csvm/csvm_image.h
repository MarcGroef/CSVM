#ifndef CSVM_IMAGE_H
#define CSVM_IMAGE_H

#include <iostream>
#include <cstdlib>
#include <vector>

#include <lodepng/lodepng.h>

using namespace std;
using namespace lodepng;


namespace csvm{
   
   enum ImageFormat{
      CSVM_IMAGE_EMPTY=0,
      CSVM_IMAGE_UCHAR_RGBA=1,
      
      
   };
   
   class Image{
      vector<unsigned char> image;
      unsigned int width;
      unsigned int height;
      ImageFormat format;
      
   public:
      Image();
      Image(string filename);
      Image(Image* im);
      void loadImage(string filename);
      vector<unsigned char> getImage();
      void exportImage(string filename);
      unsigned char getPixel(int x,int y,int channel);
      void setPixel(int x,int y,int channel,unsigned char value);
      int getWidth();
      int getHeight();
      ImageFormat getFormat();
      Image clone();
      
   };
   
}
#endif