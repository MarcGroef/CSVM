#include <csvm/csvm_patch.h>

/* This class contains functionality for the "Patch". A patch is basically a pointer to an Image with coordinates and a patch-size.
 * This allows quick passings of a Patch.
 * 
 */

using namespace std;
using namespace csvm;



//constructor
Patch::Patch(Image* source, int x, int y, int width, int height){
   this->isSet = true;
   offsetX = x;
   offsetY = y;
   this->width = width;
   this->height = height;
   this->source = source;
   //cout << "Patch from " << x << ", " << y << ", with = " << width << ", height = " << height << endl;
   analyze();
   this->square = calculateSquare();
}

//empty constructor
Patch::Patch(){
   isSet = false;
   source = NULL;
}

Patch::Patch(Image* source){
   isSet = false;
   this->source = source;
   analyze();
}

void Patch::analyze(){
   mean = 0;
   
   for(size_t xIdx = 0; xIdx != width; ++xIdx){
      for(size_t yIdx = 0; yIdx != height; ++yIdx){
         mean += getPixel(xIdx, yIdx, 0);
      }
   }
   mean /= (height * width);
   stddev = 0;
   
   for(size_t xIdx = 0; xIdx != width; ++xIdx){
      for(size_t yIdx = 0; yIdx != height; ++yIdx){
         stddev += (mean - getPixel(xIdx, yIdx, 0)) * (mean - getPixel(xIdx, yIdx, 0));
      }
   }
   stddev /= (height * width);
   stddev = sqrt(stddev + 0.001);
}

void Patch::setArea(int x,int y,int width,int height){
   offsetX = x;
   offsetY = y;
   this->width = width;
   this->height = height;
}

int Patch::calculateSquare(){
	bool trueMiddelX = 0;
   bool trueMiddelY = 0;

   int maxOffSetWidth = source->getWidth()-width;
   int middelOffSetWidth = maxOffSetWidth/2;

   int maxOffSetHeigth = source->getHeight()-height;
   int middelOffSetHeigth = maxOffSetHeigth/2;
   if(maxOffSetWidth%2==0)
      trueMiddelX=1;

   if(maxOffSetHeigth%2==0)
      trueMiddelY=1;
   
   if(trueMiddelX && trueMiddelY){
      if(offsetX < middelOffSetWidth && offsetY < middelOffSetHeigth)//topleft
         return 0;
      if(offsetX >= middelOffSetWidth && offsetY < middelOffSetHeigth)//topright
         return 1;
      if(offsetX < middelOffSetWidth && offsetY >= middelOffSetHeigth)//bottom-left
         return 2;
      if(offsetX >= middelOffSetWidth && offsetY >= middelOffSetHeigth)//bottom-right
         return 3;
   }
   if(!trueMiddelX && !trueMiddelY){
      if(offsetX <= middelOffSetWidth && offsetY <= middelOffSetHeigth)//topleft
         return 0;
      if(offsetX > middelOffSetWidth && offsetY <= middelOffSetHeigth)//topright
         return 1;
      if(offsetX <= middelOffSetWidth && offsetY > middelOffSetHeigth)//bottom-left
         return 2;
      if(offsetX > middelOffSetWidth && offsetY > middelOffSetHeigth)//bottom-right
         return 3;
   }
   //cout << offsetX << ", " << offsetY << ", " << square << endl;
   return -1;
}

//get pixel from image at location in patch

unsigned char Patch::getPixel(unsigned int x, unsigned int y,int channel){
   if(x > width || y > height)
      cout << "Patch get picel out of bounds!\n";
   return source->getPixel(offsetX+x,offsetY+y,channel);
}

//set pixel in image at location in patch
void Patch::setPixel(int x,int y,int channel,unsigned char value){
   source->setPixel(offsetX+x,offsetY+y,channel,value);
}

//get a pointer to the image
Image* Patch::getSource(){
   return source;
}

//get width of the patch
int Patch::getWidth(){
   return width;
}

//get top-left x-location of patch
int Patch::getX(){
   return offsetX;
}

//get top-left y-location of patch
int Patch::getY(){
   return offsetY;
}

//get which square patch is located in
unsigned int Patch::getSquareId(){
   return square;
}

//check whether 2 patches are equal
bool Patch::equals(Patch p){
    return (p.getSource() == source) && (offsetX == p.getX()) && ( offsetY == p.getY());
}

//get height of the patch
int Patch::getHeight(){
   return height;
}

//get pixel from patch, but converted to grey pixels
float Patch::getGreyPixel(int x, int y){
   unsigned char val = source->getGreyPixel(offsetX+x,offsetY+y);
   val = val > 255 ? 255 : val;
   return (float)val;
   return (float)(val - mean)/ stddev;
}

//get label from source-image
string Patch::getLabel(){
   return source->getLabel();
}

//get label-id (label index) from source image
unsigned int Patch::getLabelId(){
   return source->getLabelId();
}
