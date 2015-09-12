#include <csvm/csvm_image_scanner.h>

using namespace csvm;
using namespace std;

ImageScanner::ImageScanner(int wSize,int nPatches){
   winSize = wSize;
   this->nPatches = nPatches;

}

void ImageScanner::setImage(string filename){
   image.loadImage(filename);
   imageDir = filename;

}





void ImageScanner::scanImage(){
   //for(int patch=0;patch<nPatches;patch++){



   //}
   HOGDescriptor hog(9,4,4,32,16);
   v_hogValues = hog.getHOG(image,0);
   cout << "done Imagescan. "<< v_hogValues.size() << " HOGs obtained \n";
}
