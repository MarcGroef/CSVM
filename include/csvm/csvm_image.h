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
      CSVM_IMAGE_EMPTY = 0,
      CSVM_IMAGE_UCHAR_RGBA = 1,
      CSVM_IMAGE_UCHAR_RGB = 2,
      
   };
   
   class Image{
      vector<unsigned char> image;
      unsigned int width;
      unsigned int height;
      ImageFormat format;
      
      bool hasLabel;
      string label;
      
   public:
      Image();
      Image(int width,int height,ImageFormat f);
      Image(string filename);
      Image(Image* im);
      Image(Image* ROI_source,int ROI_x,int ROI_y,int ROI_width,int ROI_height);
      void loadImage(string filename);
      vector<unsigned char> getImage();
      void exportImage(string filename);
      unsigned char getPixel(int x,int y,int channel);
      void setPixel(int x,int y,int channel,unsigned char value);
      int getWidth();
      int getHeight();
      ImageFormat getFormat();
      Image clone();
      Image getROI(int x,int y,int RegionWidth,int RegionHeight);
      bool isLabeled();
      string getLabel();
      void setLabel(string l);
      void convertTo(ImageFormat f);
     
   };
   
}
#endif