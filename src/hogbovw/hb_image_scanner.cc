#include <hogbovw/hb_image_scanner.h>

using namespace hogbovw;
using namespace cv;
using namespace std;

ImageScanner::ImageScanner(int wSize,int nPatches){
   winSize = wSize;
   this->nPatches = nPatches;

}

void ImageScanner::setImage(string filename){
   image = imread(filename,0);
   imageDir = filename;
   if (!image.data)
      cout << "Failed loading image " << filename << "\n";
   else
      cout << "Image "<< filename <<" loaded succesfully. " << image.cols << "x" << image.rows << "pixels\n";

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
   //for(int patch=0;patch<nPatches;patch++){



   //}
   HOGDescriptor hog(9,4,4,32,16);
   
   v_hogValues = hog.getHOG(image);
   cout << "done Imagescan. "<< v_hogValues.size() << " HOGs obtained \n";
}
