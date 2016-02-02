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
}

//empty constructor
Patch::Patch(){
   isSet = false;
   source = NULL;
}

Patch::Patch(Image* source){
   isSet = false;
   this->source = source;
}

void Patch::setArea(int x,int y,int width,int height){
   offsetX = x;
   offsetY = y;
   this->width = width;
   this->height = height;
}

//get pixel from image at location in patch

unsigned char Patch::getPixel(int x,int y,int channel){
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

//check whether 2 patches are equal
bool Patch::equals(Patch p){
    return (p.getSource() == source) && (offsetX == p.getX()) && ( offsetY == p.getY());
}

//get height of the patch
int Patch::getHeight(){
   return height;
}

//get pixel from patch, but converted to grey pixels
unsigned char Patch::getGreyPixel(int x, int y){
   unsigned char val = source->getGreyPixel(offsetX+x,offsetY+y);
   
    return val > 255 ? 255 : val;
}

//get label from source-image
string Patch::getLabel(){
   return source->getLabel();
}

//get label-id (label index) from source image
unsigned int Patch::getLabelId(){
   return source->getLabelId();
}