#include <csvm/csvm_image_scanner.h>

using namespace csvm;
using namespace std;

ImageScanner::ImageScanner(){
   srand(time(NULL));
}

vector<Patch> ImageScanner::scanImage(Image* image,unsigned int patchWidth,unsigned int patchHeight,unsigned int xStride,unsigned int yStride){
   vector<Patch> patches((image->getWidth()-patchWidth)*(image->getHeight()-patchHeight));
   
   unsigned int scanWidth = image->getWidth() - patchWidth;
   unsigned int scanHeight = image->getHeight() - patchHeight;
   
   unsigned int patchesTaken = 0;
   
   for(size_t x = 0; x < scanWidth; ++x){
      for(size_t y = 0; y < scanHeight; ++y){
         patches[patchesTaken] = Patch(image, x, y, patchWidth, patchHeight);
      }
   }
   
   return patches;
}
      


vector<Patch> ImageScanner::getRandomPatches(Image* image, unsigned int nPatches,unsigned int patchWidth, unsigned int patchHeight){
   vector<Patch> patches(nPatches);
   
   int scanWidth = image->getWidth() - patchWidth;
   int scanHeight = image->getHeight() - patchHeight;
   
   for(size_t idx = 0; idx < nPatches; ++idx){
      patches[idx] = Patch(image,rand()%scanWidth,rand()%scanHeight,patchWidth,patchHeight);
   }
   return patches;
}