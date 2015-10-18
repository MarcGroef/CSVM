#include <csvm/csvm_patch.h>

using namespace std;
using namespace csvm;

Patch::Patch(Image* source, int x, int y, int width,int height){
   this->isSet = true;
   offsetX = x;
   offsetY = y;
   this->width = width;
   this->height = height;
   this->source = source;
}

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

unsigned char Patch::getPixel(int x,int y,int channel){
   return source->getPixel(offsetX+x,offsetY+y,channel);
}

void Patch::setPixel(int x,int y,int channel,unsigned char value){
   source->setPixel(offsetX+x,offsetY+y,channel,value);
}

Image* Patch::getSource(){
   return source;
}

int Patch::getWidth(){
   return width;
}

int Patch::getX(){
   return offsetX;
}

int Patch::getY(){
   return offsetY;
}

bool Patch::equals(Patch p){
    return (p.getSource() == source) && (offsetX == p.getX()) && ( offsetY == p.getY());
}
int Patch::getHeight(){
   return height;
}

unsigned char Patch::getGreyPixel(int x, int y){
   double val = source->getGreyPixel(offsetX+x,offsetY+y);
   
    return val > 255 ? 255 : val;
}

string Patch::getLabel(){
   return source->getLabel();
}