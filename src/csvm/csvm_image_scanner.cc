#include <csvm/csvm_image_scanner.h>

using namespace csvm;
using namespace std;

/* The image-scanner is responsible for extracting patches from images.
 * It can extract patches in a convolutional manner, using the patch-specifications in the settings-file.
 * 
 * It is also able to extract one patch from a given image, at a particular location.
 * 
 * An last, but not least, extract a patch at a random location of an image.
 * 
 * 
 * 
 */

ImageScanner::ImageScanner(){
   srand(time(NULL));
}

//set settings (done by classifier)
void ImageScanner::setSettings(ImageScannerSettings set){
  settings = set;
}

//get all patches from image.
vector<Patch> ImageScanner::scanImage(Image* image){
   vector<Patch> patches;
   
   unsigned int scanWidth = image->getWidth() - settings.patchWidth;
   unsigned int scanHeight = image->getHeight() - settings.patchHeight;
   
   
   if(scanWidth == 0 || scanHeight == 0){
      return vector<Patch>(1,Patch(image, 0, 0, settings.patchWidth,settings.patchHeight));
   }

   for(size_t x = 0; x + settings.patchWidth  <= image->getWidth(); x += settings.stride){
      for(size_t y = 0; y + settings.patchHeight  <= image->getHeight(); y += settings.stride){
         
         patches.push_back(Patch(image, x, y, settings.patchWidth, settings.patchHeight));
         
      }
   }
   //cout << "Patch width = " << patches[patchesTaken - 1].getWidth() << ", height = " << patches[patchesTaken - 1].getHeight() << endl;;
   return patches;
}

//get patch at particular location.
Patch ImageScanner::getPatchAt(Image* image, unsigned int x, unsigned int y){
   if(x + settings.patchWidth > image->getWidth() || y + settings.patchHeight > image->getHeight()){
      cout << "Image scanner WARNING!! Requested patch at " << x << ", " << y << " is out of bounds!\n";
      exit(0);
   }
   
   return Patch(image, x, y, settings.patchWidth, settings.patchHeight);
}

//get random patch from image
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

void ImageScanner::setScannerStride(unsigned int stride){
	settings.stride = stride;
	std::cout << "The stride of the Image scanner is set to " << stride << std::endl;
}

unsigned int ImageScanner::getScannerStride(){
   return settings.stride;
}
