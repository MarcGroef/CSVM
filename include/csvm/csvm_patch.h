#ifndef CSVM_PATCH_H
#define CSVM_PATCH_H

/* This class contains functionality for the "Patch". A patch is basically a pointer to an Image with coordinates and a patch-size.
 * This allows quick passings of a Patch.
 * 
 */

#include "csvm_image.h"
#include <cstdlib>

using namespace std;

namespace csvm{

   class Patch{
      Image* source;
      int offsetX,offsetY;
      int width,height;
      bool isSet;
   public:
      bool debugOut, normalOut;
      Patch(Image* source, int x, int y, int width,int height);
      Patch();
      Patch(Image* source);
      int getX();
      int getY();
      void setArea(int x,int y,int width,int height);
      unsigned char getPixel(int x,int y,int channel);
      void setPixel(int x,int y,int channel,unsigned char value);
      int getWidth();
      int getHeight();
      unsigned char getGreyPixel(int x,int y);
      string getLabel();
      unsigned int getLabelId();
      bool equals(Patch p);
      Image* getSource();
   };
}
#endif