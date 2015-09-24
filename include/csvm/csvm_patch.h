#ifndef CSVM_PATCH_H
#define CSVM_PATCH_H

using namespace std;

namespace csvm{

   class Patch{
      Image* source;
      int offsetX,offsetY;
      int width,height;
      bool isSet;
   public:
      Patch(Image* source, int x, int y, int width,int height);
      Patch();
      Patch(Image* source);
      void setArea(int x,int y,int width,int height);
      unsigned char getPixel(int x,int y,int channel);
      void setPixel(int x,int y,int channel,unsigned char value);
      
   }
}
#endif