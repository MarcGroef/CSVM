#include <csvm/csvm_image.h>

using namespace std;
using namespace lodepng;


namespace csvm{
   
   Image::Image(){
      format = CSVM_IMAGE_EMPTY;  
   }
   
   Image::Image(string filename){
      loadImage(filename);
   }
   
   Image::Image(Image* im){
      image = im->getImage();
      width = im->getWidth();
      height = im->getHeight();
      format = im->getFormat();
   }
   
   void Image::loadImage(string filename){
      string png = ".png";
      
      if(filename.length() > png.length() && 0 == filename.compare(filename.length() - png.length(), png.length(), png)){
         unsigned int error = decode(image,width,height,filename);
         if(error)
            cout << "csvm::Image::loadImage(std::string) Error: " << lodepng_error_text(error) << "\n";
         else{
            format = CSVM_IMAGE_UCHAR_RGBA;
         }
      }else
         cout << "csvm::Image::loadImage(std::string) Error: Given filename " << filename << " has an unsupported extention.\nSupported extentions are: .png\n";
        
      
      
   }
   
   unsigned char Image::getPixel(int x,int y,int channel){
      switch(format){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::getPixel() Warning! Image not set! returning 0\n";
            return 0;
         case CSVM_IMAGE_UCHAR_RGBA:
            return image[(width * y * 4) + (x * 4) + channel];
         
         
      }
   }
   
   void Image::setPixel(int x,int y,int channel,unsigned char value){
      switch(format){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::setPixel() Warning! Image not set!\n";
            return;
         case CSVM_IMAGE_UCHAR_RGBA:
            image[(width * y * 4) + (x * 4) + channel] = value;
         break;
      }      
   }
   
   vector<unsigned char> Image::getImage(){
      return image;
   }
   
   void Image::exportImage(string filename){
      string png = ".png";
      
      if(filename.length() > png.length() && 0 == filename.compare(filename.length() - png.length(), png.length(), png)){  //if so, write a png file
         unsigned int error = encode(filename,image,width,height);
         if(error)
            cout << "csvm::Image::exportImage(std::string) Error: " << lodepng_error_text(error) << "\n";
         else{
            format = CSVM_IMAGE_UCHAR_RGBA;
         }
      }else
         cout << "csvm::Image::exportImage(std::string) Error: Given filename " << filename << " has an unsupported extention.\nSupported extentions are: .png\n";
        
   }
   
   int Image::getWidth(){
      return width;
   }
      
   int Image::getHeight(){
      return height;
   }
   
   ImageFormat Image::getFormat(){
      return format;
   }
   
   Image Image::clone(){
      Image clone(this);
      return clone;
   }
   
}



