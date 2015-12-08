#include <csvm/csvm_image_scanner.h>

using namespace csvm;
using namespace std;

ImageScanner::ImageScanner(){
   srand(time(NULL));
}

void ImageScanner::setSettings(ImageScannerSettings set){
  settings = set;
}

vector<Patch> ImageScanner::scanImage(Image* image){
   vector<Patch> patches;
   
   unsigned int scanWidth = image->getWidth() - settings.patchWidth;
   unsigned int scanHeight = image->getHeight() - settings.patchHeight;
   
   
   if(scanWidth == 0 || scanHeight == 0){
      return vector<Patch>(1,Patch(image, 0, 0, settings.patchWidth,settings.patchHeight));
   }

   unsigned int quadrantSize = image->getWidth()/2;
   //cout << "quadrant size = " << quadrantSize << endl;
   for(size_t x = 0; x + settings.patchWidth  <= scanWidth; x += settings.stride){
      for(size_t y = 0; y + settings.patchHeight  <= scanHeight; y += settings.stride){
         
         patches.push_back(Patch(image, x, y, settings.patchWidth, settings.patchHeight));
         
      }
   }
   //cout << "Patch width = " << patches[patchesTaken - 1].getWidth() << ", height = " << patches[patchesTaken - 1].getHeight() << endl;;
   return patches;
}
      


Patch ImageScanner::getRandomPatch(Image* image){
   
   //cout << "imscan\n";
   int scanWidth = image->getWidth() - settings.patchWidth;
   int scanHeight = image->getHeight() - settings.patchHeight;
  
   if(scanWidth == 0 || scanHeight == 0){
      return Patch(image, 0, 0, settings.patchWidth,settings.patchHeight);
   }
   vector<Patch> patches(settings.nRandomPatches);
   
  
   return Patch(image,(rand() % scanWidth), (rand() % scanHeight),settings.patchWidth,settings.patchHeight);
}