//Copyright Marc Groefsema (c) 2015

#include <hb_hog_descriptor.h>

using namespace opencv;
using namespace std;
using namespace hogbovw;

HOGDescriptor::HOGDescriptor();

vector<float> HOGDescriptor::getHOG(Mat image){
   Mat gx,gy;
   gx=image.clone();
   gy=image.clone();
   
   int imWidth = image.cols;
   int imHeight = image.rows;
   
   
}

