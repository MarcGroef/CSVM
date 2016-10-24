#include <csvm/csvm_interpolator.h>
#include <cmath>

using namespace std;
using namespace csvm;

/* BiCubic interpolation.
 * 
 * 
 * 
 * */

float cubicInterpolate (float p[4], float x) {
   return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));
}

float bicubicInterpolate (float p[4][4], float x, float y) {
   float arr[4];
   arr[0] = cubicInterpolate(p[0], y);
   arr[1] = cubicInterpolate(p[1], y);
   arr[2] = cubicInterpolate(p[2], y);
   arr[3] = cubicInterpolate(p[3], y);
   return cubicInterpolate(arr, x);
}

Image Interpolator::interpolate_bicubic(Image& im,unsigned int newWidth,unsigned int newHeight){

   Image im2(newWidth, newHeight, im.getFormat());
   unsigned int oldWidth = im.getWidth();
   unsigned int oldHeight = im.getHeight();   

   int x,y;
   float dx,dy;
   float tx,ty;

   tx = (float) oldWidth / newWidth ;
   ty = (float) oldHeight / newHeight;
   
   unsigned int nChannels = im.getNChannels();
   
   float window[4][4];

   for(size_t nxIdx = 0; nxIdx != newWidth; ++nxIdx){
      for(size_t nyIdx = 0; nyIdx != newHeight; ++nyIdx){
   
         x = (int)(tx * nxIdx);
         y = (int)(ty * nyIdx);
         
         dx = tx * nxIdx - x;
         dy = ty * nyIdx - y;
         


         for(size_t chIdx = 0; chIdx < nChannels; chIdx++){
            for(size_t winXIdx = 0; winXIdx != 4; ++winXIdx){
               for(size_t winYIdx = 0; winYIdx != 4; ++winYIdx){
                  
                  window[winXIdx][winYIdx] = (float)(im.getPixel((x + winXIdx >= oldWidth ? oldWidth - 1 : x + winXIdx),(y + winYIdx >= oldHeight ? oldHeight - 1 : y + winYIdx),chIdx));
               }
            }
            float value = bicubicInterpolate(window,dx,dy);
            im2.setPixel(nxIdx, nyIdx, chIdx, (unsigned char)(value > 255 ? 255 : (value < 0 ? 0 : value)));
            
         }
      }
   }
   return im2;   
}

