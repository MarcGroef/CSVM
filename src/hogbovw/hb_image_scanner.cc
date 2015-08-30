//copyright Marc Groefsema (c) 2015

#include <hogbovw/hb_image_scanner.h>

using namespace hogbovw;
using namespace cv;
using namespace std;

ImageScanner::ImageScanner(int wSize,int nPatches){
   winSize = wSize;
   this->nPatches = nPatches;

   hog
}

void ImageScanner::setImage(string filename){
   image = imread(filename,CV_LOAD_IMAGE_COLOR);
   imageDir = filename;
   if (!image.data)
      cout << "Failed loading image " << filename << "\n";
   else
      cout << "Image "<< filename <<" loaded succesfully.\n";

}

void ImageScanner::showImage(){
   if(!image.data){
     cout << "There is no image loaded to show..\n";
   }else{
     namedWindow("ImageScanner: Image loaded",WINDOW_AUTOSIZE);
     imshow("ImageScanner: Image loaded",image);
     waitKey(0);
   }
}

void ImageScanner::scanImage(){
   for(int patch=0;patch<nPatches;patch++){



   }

}
