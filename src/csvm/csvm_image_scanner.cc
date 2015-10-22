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
   
   int scanWidth = image->getWidth() - settings.patchWidth;
   int scanHeight = image->getHeight() - settings.patchHeight;
   unsigned int random;
   
   vector<unsigned int> historyX(settings.nRandomPatches,-1);
   vector<unsigned int> historyY(settings.nRandomPatches,-1);
   unsigned int xsFound = 0;
   unsigned int ysFound = 0;
  
  
   //cout << "scanWidth = " << scanWidth << endl;
   //cout << "scanHeight = " << scanHeight << endl;
   
   for(size_t idx = 0; idx < settings.nRandomPatches; ++idx){
      bool present = true;
      for( random = rand() % scanWidth; present; random = rand() % scanWidth){
         //cout << "radnom is now " << random << endl;
         present = false;
         for(size_t ch = 0; ch < xsFound; ++ch){
            
            if(historyY[ch] == random){
               present = true;
               //cout << random << "is present in x.." << endl;
               break;
            }
         }
         
         if(!present)
            historyX[xsFound++] = random;
      }
      //cout << "ys turn\n";
      present = true;
      for( random = (rand() % scanHeight); present; random = (rand() % scanHeight)){
         present = false;
         //cout << "radnom is now " << random << endl;
         for(size_t ch = 0; ch < ysFound; ++ch){
            if(historyY[ch] == random){
               present = true;
               //cout << random << "is present in y.." << endl;
               break;
            }
         }
         if(!present)
            historyY[ysFound++] = random;
      }
      //cout << "generated combo: " << historyX[xsFound - 1] << ", " << historyY[ysFound - 1] << endl;
   }
   
   
   for(size_t idx = 0; idx < settings.nRandomPatches; ++idx){
      //cout << "making patch at " << historyX[idx] << ", " << historyY[idx] << endl;
      patches[idx] = Patch(image,historyX[idx], historyY[idx],settings.patchWidth,settings.patchHeight);
   }
   return patches;
}