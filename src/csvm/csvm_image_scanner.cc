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
   unsigned int nPatches = ((image->getWidth()-settings.patchWidth)/settings.stride) * ((image->getHeight()-settings.patchHeight)/settings.stride);
   vector<Patch> patches;//(nPatches);
   
   unsigned int scanWidth = image->getWidth() - settings.patchWidth;
   unsigned int scanHeight = image->getHeight() - settings.patchHeight;
   
   unsigned int patchesTaken = 0;
   
   if(scanWidth == 0 || scanHeight == 0){
      return vector<Patch>(1,Patch(image, 0, 0, settings.patchWidth,settings.patchHeight));
   }
   
   //for(size_t x = 0; x < scanWidth; x += settings.stride){
      //for(size_t y = 0; y < scanHeight; y += settings.stride){
   //cout << "Image width = " << image->getWidth() << endl;;
   for(size_t x = 0; x + settings.patchWidth  <= image->getWidth(); x += settings.stride){
      for(size_t y = 0; y + settings.patchHeight  <= image->getHeight(); y += settings.stride){
         
         //cout << "x: " << x << " till " << x+settings.patchWidth << endl;
         //patches[patchesTaken] = Patch(image, x, y, settings.patchWidth, settings.patchHeight);
         patches.push_back(Patch(image, x, y, settings.patchWidth, settings.patchHeight));
         //++patchesTaken;
      }
   }
   //cout << "Patch width = " << patches[patchesTaken - 1].getWidth() << ", height = " << patches[patchesTaken - 1].getHeight() << endl;;
   return patches;
}
      


vector<Patch> ImageScanner::getRandomPatches(Image* image){
   
   //cout << "imscan\n";
   int scanWidth = image->getWidth() - settings.patchWidth;
   int scanHeight = image->getHeight() - settings.patchHeight;
  
   if(scanWidth == 0 || scanHeight == 0){
      return vector<Patch>(1,Patch(image, 0, 0, settings.patchWidth,settings.patchHeight));
   }
   vector<Patch> patches(settings.nRandomPatches);
   
   for(size_t idx = 0; idx < settings.nRandomPatches; ++idx){
      //cout << "making patch at " << historyX[idx] << ", " << historyY[idx] << endl;
      patches[idx] = Patch(image,(rand() % scanWidth), (rand() % scanHeight),settings.patchWidth,settings.patchHeight);
   }
   return patches;
}