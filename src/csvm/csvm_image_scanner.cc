#include <csvm/csvm_image_scanner.h>

using namespace csvm;
using namespace std;

ImageScanner::ImageScanner(){
   srand(time(NULL));
}

void ImageScanner::setSettings(ImageScannerSettings set){
  settings = set;
}

vector< vector<Patch> > ImageScanner::scanImage(Image* image){
   vector< vector<Patch> > patches(4);
   
   unsigned int scanWidth = image->getWidth() - settings.patchWidth;
   unsigned int scanHeight = image->getHeight() - settings.patchHeight;
   
   
   if(scanWidth == 0 || scanHeight == 0){
      return vector< vector<Patch> >(1,vector<Patch>(1,Patch(image, 0, 0, settings.patchWidth,settings.patchHeight)));
   }

   unsigned int quadrantSize = image->getWidth()/2;
   //cout << "quadrant size = " << quadrantSize << endl;
   for(size_t xQuad = 0; xQuad < 2; ++xQuad){
      for(size_t yQuad = 0; yQuad < 2; ++yQuad){
         for(size_t x = xQuad*quadrantSize; x + settings.patchWidth  <= (xQuad + 1) * quadrantSize; x += settings.stride){
            for(size_t y = yQuad * quadrantSize; y + settings.patchHeight  <= (yQuad + 1) * quadrantSize; y += settings.stride){
               
               patches[xQuad * 2 + yQuad].push_back(Patch(image, x, y, settings.patchWidth, settings.patchHeight));
               
            }
         }
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