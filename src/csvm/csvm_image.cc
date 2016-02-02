#include <csvm/csvm_image.h>

using namespace std;
using namespace lodepng;

/* The image class, contains all general functionality to do things with an image.
 * One can read and write them to/from PNG (thanks to LodePNG),
 * convert them to different memory layouts(Grey/RGB, etc), 
 * 
 * 
 * 
 * 
 * 
 */
namespace csvm{
   
	
	//empty constructor
   Image::Image(){
      format = CSVM_IMAGE_EMPTY;  
   }
   
   //Make an image by loading it from a file. (PNG atm)
   Image::Image(string filename){
      loadImage(filename);
   }
   
   //copy constructor
   Image::Image(Image* im){
      image = im->getImage();
      width = im->getWidth();
      height = im->getHeight();
      format = im->getFormat();
   }
   
   //empty constructor with future specifications for image
   Image::Image(int width,int height,ImageFormat f){
      this->width = width;
      this->height = height;
      this->format = f;
      
      switch(format){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::constructor(int,int,ImageFormat): Warning! Trying to allocate empty image. Leaving Image empty..\n";
            return;
         case CSVM_IMAGE_UCHAR_RGB:
            image.resize(3*this->width*this->height);
            return;
         case CSVM_IMAGE_UCHAR_RGBA:
            image.resize(4*this->width*this->height);
            return;
         case CSVM_IMAGE_UCHAR_GREY:
            image.resize(this->width*this->height);
            return;
      }
   }
   
   //ROI constructor
   Image::Image(Image* ROI_source,unsigned int ROI_x,unsigned int ROI_y,unsigned int ROI_width,unsigned int ROI_height){
      width = ROI_width;
      height = ROI_height;
      format = ROI_source->getFormat();
      
      if(ROI_x+ROI_width > ROI_source->getWidth() || ROI_y + ROI_height > ROI_source->getHeight()){
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
            for(size_t x = 0; x < ROI_width; x++){
               for(size_t y = 0; y < ROI_height;y++){
                  for(size_t ch = 0; ch < 4; ch++){
                     setPixel(x,y,ch,ROI_source->getPixel(ROI_x+x,ROI_y+y,ch));
                  }
               }
            }
            break;
         case CSVM_IMAGE_UCHAR_RGB:
            image.resize(ROI_height*ROI_width*3);
            for(size_t x = 0; x < ROI_width; x++){
               for(size_t y = 0; y < ROI_height;y++){
                  for(size_t ch = 0; ch < 3; ch++){
                     setPixel(x,y,ch,ROI_source->getPixel(ROI_x+x,ROI_y+y,ch));
                  }
               }
            }
            break;
         case CSVM_IMAGE_UCHAR_GREY:
            image.resize(ROI_height*ROI_width);
            for(size_t x = 0; x < ROI_width; x++){
               for(size_t y = 0; y < ROI_height; y++){
                  setPixel(x,y,0,ROI_source->getPixel(ROI_x+x,ROI_y+y,0));
               }
            }
            break;
      }
      
   }
   
   //get value directly from byte-array 
   unsigned char Image::getPixelAtIdx(unsigned int idx){
      return image[idx];
   }
   
   //get the byte-array itself
   vector<unsigned char>& Image::getData(){
      return image;
   }
   
   //get number of channels of the image
   unsigned int Image::getNChannels(){
      switch(format){
         case CSVM_IMAGE_EMPTY: return 0;
         case CSVM_IMAGE_UCHAR_RGBA: return 4;
         case CSVM_IMAGE_UCHAR_RGB: return 3;
         case CSVM_IMAGE_UCHAR_GREY: return 1;
      }
      return -1;
   }
   
   //load image (currently from PNG)
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
   
   //get pixel value at <x,y,channel>
   unsigned char Image::getPixel(unsigned int x,unsigned int y,unsigned int channel){
      if( x < 0 || x >= width || y < 0 || y >= width){
         cout << "csvm::Image::getPixel() Warning! coordinate out of bounds! : " << x << ", " << y <<" returning 0\n";
         return 0;
      }
      switch(format){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::getPixel() Warning! Image not set! returning 0\n";
            return -1;
         case CSVM_IMAGE_UCHAR_RGBA:
            return image[(width * y * 4) + (x * 4) + channel];
         case CSVM_IMAGE_UCHAR_RGB:
            return image[(width * y * 3) + (x * 3) + channel];
         case CSVM_IMAGE_UCHAR_GREY:
            return image[(width * y) + (x)];
      }
      return 0;
   }
   
   //set pixel at <x,y,channel>
   void Image::setPixel(unsigned int x,unsigned int y,unsigned int channel,unsigned char value){
      switch(format){
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::setPixel() Warning! Image not set!\n";
            return;
         case CSVM_IMAGE_UCHAR_RGBA:
            image[(width * y * 4) + (x * 4) + channel] = value;
         break;
         case CSVM_IMAGE_UCHAR_RGB:
            image[(width * y * 3) + (x * 3) + channel] = value;
            break;
         case CSVM_IMAGE_UCHAR_GREY:
            image[width * y + x] = value;
            break;
      }      
   }
   
   //get byte array
   vector<unsigned char> Image::getImage(){
      return image;
   }
   
   //export image (currently to PNG, but you should have the ".png" extension in 'filename', so it knows you want png) 
   void Image::exportImage(string filename){
      string png = ".png";
      unsigned int error;
      if(filename.length() > png.length() && 0 == filename.compare(filename.length() - png.length(), png.length(), png)){  //if so, write a png file
         switch(format){
            case CSVM_IMAGE_UCHAR_RGBA:
            error = encode(filename,image,width,height);
            
            if(error)
               cout << "csvm::Image::exportImage(std::string) Error: " << lodepng_error_text(error) << "\n";
            break;
            case CSVM_IMAGE_UCHAR_RGB:
               error = encode(filename,image,width,height,LCT_RGB, 8);
               cout << "warning! not yet impl. (export im)\n";
               break;
            case CSVM_IMAGE_UCHAR_GREY:
               //cout << "Export grey image. warning! not yet implementer!\n";
               error = encode(filename,image,width,height,LCT_GREY, 8);
               break;
            case CSVM_IMAGE_EMPTY:
               break;
         }
      }else
         cout << "csvm::Image::exportImage(std::string) Error: Given filename " << filename << " has an unsupported extention.\nSupported extentions are: .png\n";
        
   }
   
   //get width of image
   unsigned int Image::getWidth(){
      return width;
   }
   
   //get height of image
   unsigned int Image::getHeight(){
      return height;
   }
   
   
   //get format of image, (grey, rgb, etc)
   ImageFormat Image::getFormat(){
      return format;
   }
   
   //clone the image
   Image Image::clone(){
      Image clone(this);
      return clone;
   }
   
   //get Region of interest from the image
   Image Image::getROI(unsigned int x,unsigned int y,unsigned int regionWidth,unsigned int regionHeight){
      Image roi(this,x,y,regionWidth,regionHeight);
      return roi;
   }
   
   //check whether image has a label
   bool Image::isLabeled(){
      return hasLabel;
   }
      
   //get image label
   string Image::getLabel(){
      return label;
   }
   
   //set image label
   void Image::setLabel(string label){
      this->label = label;
   }
   
   //convert to particular channel format
   Image Image::convertTo(ImageFormat f){
      Image im;
      
      if(format==f)
         return im.clone();     //nothing to do
      switch(format){
         
         case CSVM_IMAGE_EMPTY:
            cout << "csvm::Image::convert(ImageFormat) Warning! Trying to convert from an empty image. Returning..\n";
            break;
            
         case CSVM_IMAGE_UCHAR_RGB:  
            switch(f){
               case CSVM_IMAGE_EMPTY:
                  cout << "csvm::Image::convert(ImageFormat) Warning! Trying to convert to an empty image. Returning..\n";
                  break;
               case CSVM_IMAGE_UCHAR_RGB: //case already handled above
                  break;
               case CSVM_IMAGE_UCHAR_RGBA:
                  im = UCHAR_RGB2UCHAR_RGBA();
                  break;
               case CSVM_IMAGE_UCHAR_GREY:
                  cout << "RGB to GREY: not yet implemented!\n";
                  break;
            }
            break;
         case CSVM_IMAGE_UCHAR_RGBA:
            switch(f){
               case CSVM_IMAGE_EMPTY:
                  cout << "csvm::Image::convert(ImageFormat) Warning! Trying to convert to an empty image. Returning..\n";
                  break;
               case CSVM_IMAGE_UCHAR_RGB: 
                  im = UCHAR_RGBA2UCHAR_RGB();
                  break;
               case CSVM_IMAGE_UCHAR_RGBA://case already handled above
                  break;
               case CSVM_IMAGE_UCHAR_GREY:
                  cout << "RGBA to GREY: not yet implemented!\n";
                  break;
            }
            break;
            
         case CSVM_IMAGE_UCHAR_GREY:
            cout << "image conversion from grey: not yet implemented!\n";
            break;
      }
      return im;
   }
   
   void Image::setLabelId(int id){
      labelId = id;
   }
   
   unsigned int Image::getLabelId(){
      return labelId;
   }
   
   unsigned char Image::getGreyPixel(unsigned int x,unsigned int y){
      return 0.2126 * getPixel(x,y,0) + 0.7152 * getPixel(x,y,1) + 0.0722 * getPixel(x,y,2);
   }
   
   void Image::setImageData(vector<unsigned char> data){
      image = data;
   }
   
   //------------------------------- private methods ------------------------------------//
   //Conversion between memory layouts and stuff
   
   Image Image::UCHAR_RGB2UCHAR_RGBA(){
      Image newImage(width,height,CSVM_IMAGE_UCHAR_RGBA);
      newImage.setLabel(label);
     
      for(unsigned int y = 0; y < height; y++){
         for(unsigned int x = 0; x < width; x++){
            for(int ch = 0; ch < 4;ch++){
               if(ch != 3)
                  newImage.setPixel(x,y,ch,getPixel(x,y,ch));
               else
                  newImage.setPixel(x,y,ch,255);
            }
         }
      }
      return newImage;
      
   }
   
   Image Image::UCHAR_RGBA2UCHAR_RGB(){
      Image newImage(width,height,CSVM_IMAGE_UCHAR_RGB);
      
     
      for(unsigned int y = 0; y < height; y++){
         for(unsigned int x = 0; x < width; x++){
            for(int ch = 0; ch < 3;ch++){
               newImage.setPixel(x,y,ch,getPixel(x,y,ch));
            }
         }
      }
      return newImage;
   }
   
   Image Image::UCHAR_RGB2UCHAR_GREY(){
      Image newImage(width,height,CSVM_IMAGE_UCHAR_GREY);
      unsigned char val;
      for(unsigned int y = 0; y < height; y++){
         for(unsigned int x = 0; x < width; x++){
            val = 0.299 * getPixel(x,y,0) + 0.7152 * getPixel(x,y,1) + 0.0722 * getPixel(x,y,2);
            newImage.setPixel(x,y,0,val);
            
         }
      }
      return newImage;
      
   }
   
   Image Image::UCHAR_RGBA2UCHAR_GREY(){
      Image newImage(width,height,CSVM_IMAGE_UCHAR_GREY);
      unsigned char val;
      for(unsigned int y = 0; y < height; y++){
         for(unsigned int x = 0; x < width; x++){
            val = 0.2126 * getPixel(x,y,0) + 0.7152 * getPixel(x,y,1) + 0.0722 * getPixel(x,y,2);
            newImage.setPixel(x,y,0,val);
            
         }
      }
      return newImage;
   }
}



