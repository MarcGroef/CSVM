#include <csvm/csvm_centroid.h>


//DEPRECATED
using namespace std;
using namespace csvm;

double Centroid::getDistanceSq(Feature f){
   unsigned int nDims = f.content.size();
   double sum = 0.0;
   for(size_t dIdx = 0; dIdx < nDims; ++dIdx){
      sum += (content[dIdx] - f.content[dIdx]) * (content[dIdx] - f.content[dIdx]);
   }
   return sum;
}


double Centroid::getDistanceSq(Centroid c) {
	unsigned int nDims = c.content.size();
	double sum = 0.0;
	for (size_t dIdx = 0; dIdx < nDims; ++dIdx) {
		sum += (content[dIdx] - c.content[dIdx]) * (content[dIdx] - c.content[dIdx]);
	}
	return sum;
}

void Centroid::exportToPNG(string name){
   unsigned int size = content.size() / 3;
   unsigned int width = sqrt(size);
   unsigned int height = width;
   
   if((width * width != size))
      ++height;
   cout << "imHeight = " << height << ", width = " << width << endl;
   Image im(width, height, CSVM_IMAGE_UCHAR_RGB);
   
   size_t fIdx = 0;
   for(size_t chIdx = 0; chIdx != 3; ++chIdx){
      for(size_t xIdx = 0; xIdx != width; ++xIdx){
         for(size_t yIdx = 0; yIdx != height; ++yIdx){
         
            //cout << "chIdx = " << chIdx << " xIdx = " << xIdx << ", yIdx = " << yIdx << ", value = " <<    (content[chIdx * 3 + yIdx * width + xIdx]) <<endl;
         
            im.setPixel(xIdx, yIdx, chIdx, (unsigned char) (content[chIdx * (width * height) + yIdx * width + xIdx]));
         }
      }
   }
   Interpolator ip;
   
   ip.interpolate_bicubic(im, 40, 40).exportImage(name + ".png");
}