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
   vector<Patch> patches((image->getWidth()-settings.patchWidth)*(image->getHeight()-settings.patchHeight));
   
   unsigned int scanWidth = image->getWidth() - settings.patchWidth;
   unsigned int scanHeight = image->getHeight() - settings.patchHeight;
   
   unsigned int patchesTaken = 0;
   
   for(size_t x = 0; x < scanWidth; ++x){
      for(size_t y = 0; y < scanHeight; ++y){
         patches[patchesTaken] = Patch(image, x, y, settings.patchWidth, settings.patchHeight);
         ++patchesTaken;
      }
   }
   
   return patches;
}
      


vector<Patch> ImageScanner::getRandomPatches(Image* image){
   vector<Patch> patches(settings.nRandomPatches);
   //cout << "imscan\n";
   int scanWidth = image->getWidth() - settings.patchWidth;
   int scanHeight = image->getHeight() - settings.patchHeight;
  
  
   
   
   
   for(size_t idx = 0; idx < settings.nRandomPatches; ++idx){
      //cout << "making patch at " << historyX[idx] << ", " << historyY[idx] << endl;
      patches[idx] = Patch(image,(rand() % scanWidth), (rand() % scanHeight),settings.patchWidth,settings.patchHeight);
   }
   return patches;
}