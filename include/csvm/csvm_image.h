#ifndef CSVM_IMAGE_H
#define CSVM_IMAGE_H

#include <iostream>
#include <cstdlib>
#include <vector>

#include <lodepng/lodepng.h>

using namespace std;
using namespace lodepng;


namespace csvm{
   // Image formats are CSVM_IMAGE_<TYPE>_<LAYOUT>
   // e.g. UCHAR_RGB is an image of unsigned char's, in a layout of rgbrgbrgbrgb....
   
   enum ImageFormat{
      CSVM_IMAGE_EMPTY = 0,
      CSVM_IMAGE_UCHAR_RGBA = 1,
      CSVM_IMAGE_UCHAR_RGB = 2,
      CSVM_IMAGE_UCHAR_GREY = 3,
   };
   
   class Image{
      vector<unsigned char> image;
      unsigned int width;
      unsigned int height;
      ImageFormat format;
      
      bool hasLabel;
      string label;
      unsigned int labelId;
      
      //private methods:
      Image UCHAR_RGB2UCHAR_RGBA();
      Image UCHAR_RGBA2UCHAR_RGB();
      Image UCHAR_RGB2UCHAR_GREY();
      Image UCHAR_RGBA2UCHAR_GREY();
      
      
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
      unsigned char getGreyPixel(int x, int y);
      void setPixel(int x,int y,int channel,unsigned char value);
      int getWidth();
      int getHeight();
      
      ImageFormat getFormat();
      Image clone();
      Image getROI(int x,int y,int RegionWidth,int RegionHeight);
      bool isLabeled();
      string getLabel();
      int getLabelId();
      void setLabel(string l);
      void setLabelId(int id);
      Image convertTo(ImageFormat f);
     
   };
   
}
#endif