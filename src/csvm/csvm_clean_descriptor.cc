#include <csvm/csvm_clean_descriptor.h>
#include <cmath>
using namespace std;
using namespace csvm;


//simply copy all pixels in grey-value towards a feature vector
Feature CleanDescriptor::describe(Patch p){  
   
   unsigned int imHeight = p.getHeight();
   unsigned int imWidth = p.getWidth();
   unsigned int numColours = (p.getSource()->getFormat() == CSVM_IMAGE_UCHAR_GREY ? 1 : 3)  ;
   unsigned int imSize = (imWidth * imHeight * numColours);//3);
   unsigned int chSize = imWidth * imHeight;
   
   Feature f(imSize,0);
   
   
   f.content = vector<double>(imSize,0);
   
   /*for(size_t idxX = 0; idxX < imWidth; ++idxX){
      for(size_t idxY = 0; idxY < imHeight; ++idxY){
         f.content[idxY * imWidth + idxX] = (double)(p.getGreyPixel(idxX,idxY)) ;
      }
   }*/
  
   for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
      
      double mean = 0.0;
      double stddev = 0.0;
      
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            f.content[chIdx * chSize + idxY * imWidth + idxX] = (double)(p.getPixel(idxX,idxY,chIdx)) ;
            //mean += f.content[idxY * imWidth + idxX];
            
         }
      }
      /*
   // TO DO: fix magic number 3 only appropriate when using colour, not when grayscaling
   
      mean /= (imWidth * imHeight);//3);
   //for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            
            stddev += (mean - f.content[chIdx * chSize + idxY * imWidth + idxX]) * (mean - f.content[chIdx * chSize + idxY * imWidth + idxX]);
            
         }
      }
      //}
      //cout << "var = " << stddev << ", nVals = " << (3 * imWidth * imHeight) << endl;
      stddev /= (imWidth * imHeight); //* 3);
      stddev = sqrt(stddev);
      //cout << "stddev = " << stddev << endl;
      if (stddev > 0){
      //for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
         for(size_t idxX = 0; idxX < imWidth; ++idxX){
            for(size_t idxY = 0; idxY < imHeight; ++idxY){
               f.content[chIdx * chSize + idxY * imWidth + idxX] = (f.content[chIdx * chSize + idxY * imWidth + idxX] - mean)/stddev;
               //cout << "newVal = " << f.content[idxY * imWidth + idxX] << endl;
            }
         }
      //}
      }*/
   }
   return f;
   
}  