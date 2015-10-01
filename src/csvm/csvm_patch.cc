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

int Patch::getWidth(){
   return width;
}


int Patch::getHeight(){
   return height;
}