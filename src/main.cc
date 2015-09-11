
#include <csvm/csvm_image.h>
#include <iostream>

using namespace csvm;
using namespace std;

int main(int argc,char**argv){
  /* ImageScanner is(15,100);  //window size, nPatches
   is.setImage("lenna.png");  //load image into ImageScanner using openCV
   is.showImage();            // popup a window to show the image
   
   is.scanImage();            //scan the image 

   */
   Image im("lenna.png");
   Image cl;
   cl = im.getROI(256,256,50,50);
   cl.exportImage("ROI.png");
   return 0;
}






