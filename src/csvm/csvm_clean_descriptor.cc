#include <csvm/csvm_clean_descriptor.h>
#include <cmath>

/* Put pixel intensity values in a Feature
 * 
 * Needs TODO fixes
 * 
 * 
 */


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
  double mean = 0.0;
      double stddev = 0.0;
   for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
      
      
      
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            f.content[chIdx * chSize + idxY * imWidth + idxX] = (double)(p.getPixel(idxX,idxY,chIdx)) ;
            mean += f.content[idxY * imWidth + idxX];
            
         }
      }
   }
  
   mean /= (imWidth * imHeight * numColours);//3);
   for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            
            stddev += (mean - f.content[chIdx * chSize + idxY * imWidth + idxX]) * (mean - f.content[chIdx * chSize + idxY * imWidth + idxX]);
            
         }
      }
   }
   stddev /= (imWidth * imHeight * numColours);
   stddev = sqrt(stddev);
   if (stddev > 0){
      for(size_t chIdx = 0; chIdx < numColours; ++chIdx){
         for(size_t idxX = 0; idxX < imWidth; ++idxX){
            for(size_t idxY = 0; idxY < imHeight; ++idxY){
               f.content[chIdx * chSize + idxY * imWidth + idxX] = (f.content[chIdx * chSize + idxY * imWidth + idxX] - mean)/stddev;
               
            }
         }
      }
   }
   
   return f;
   
}  