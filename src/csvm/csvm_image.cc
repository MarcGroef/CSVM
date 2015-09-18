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
   
   Image::Image(Image* ROI_source,int ROI_x,int ROI_y,int ROI_width,int ROI_height){
      width = ROI_width;
      height = ROI_height;
      format = ROI_source->getFormat();
      
      if(ROI_x+ROI_width>ROI_source->getWidth()||ROI_y+ROI_height>ROI_source->getHeight()){
         cout << "csvm::Image::ROI Constructor: Error! Region out of bounds!\n";
         exit(-1);
         return;
      }
      switch(ROI_source->getFormat()){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::ROI Constructor: Error! ROI source has no image! Exiting..\n";
            exit(-1);
            return;
         case CSVM_IMAGE_UCHAR_RGBA:
            image.resize(ROI_height*ROI_width*4);
            for(int x = 0; x < ROI_width; x++){
               for(int y = 0; y < ROI_height;y++){
                  for(int ch = 0; ch < 4; ch++){
                     setPixel(x,y,ch,ROI_source->getPixel(ROI_x+x,ROI_y+y,ch));
                  }
               }
            }
      }
      
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
            return -1;
         case CSVM_IMAGE_UCHAR_RGBA:
            return image[(width * y * 4) + (x * 4) + channel];
         
         
      }
      return 0;
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
   
   Image Image::getROI(int x,int y,int regionWidth,int regionHeight){
      Image roi(this,x,y,regionWidth,regionHeight);
      return roi;
   }
   
   bool Image::isLabeled(){
      return hasLabel;
   }
      
   string Image::getLabel(){
      return label;
   }
   
   void Image::setLabel(string label){
      this->label = label;
   }
   
}



