#include <csvm/csvm_image.h>

using namespace std;
using namespace lodepng;


namespace csvm{
   
   void Image::loadImage(string filename){
      string png = ".png";
      
      if(filename.length() > png.length() && 0 == filename.compare(filename.length() - png.length(), png.length(), png)){
         unsigned int error = decode(image,width,height,filename);
         if(error)
            cout << "csvm::Image::loadImage(std::string) Error: " << lodepng_error_text(error) << "\n";
         else{
            format = CSVM_IMAGE_RGBA;
         }
      }else
         cout << "csvm::Image::loadImage(std::string) Error: Given filename " << filename << " has an unsupported extention.\nSupported extentions are: .png\n";
        
      
      
   }
   
   vector<unsigned char> Image::getImage(){
      return image;
   }
   
   void exportImage(string fileName){
      string png = ".png";
      
      if(filename.length() > png.length() && 0 == filename.compare(filename.length() - png.length(), png.length(), png)){  //if so, write a png file
         unsigned int error = encode(image,width,height,filename);
         if(error)
            cout << "csvm::Image::exportImage(std::string) Error: " << lodepng_error_text(error) << "\n";
         else{
            format = CSVM_IMAGE_RGBA;
         }
      }else
         cout << "csvm::Image::exportImage(std::string) Error: Given filename " << filename << " has an unsupported extention.\nSupported extentions are: .png\n";
        
   }
   
   
   
}



