
#include <csvm/csvm.h>
#include <iostream>

using namespace csvm;
using namespace std;

int main(int argc,char**argv){
   //ImageScanner is(15,100);  //window size, nPatches
   //is.setImage("lenna.png");  //load image into ImageScanner
 
   
  // is.scanImage();            //scan the image 

   
   Image im("lenna.png");
  // Image cl;
   //cl = im.getROI(256,256,50,50);
   //cl.exportImage("ROI.png");
 
   return 0;
}






