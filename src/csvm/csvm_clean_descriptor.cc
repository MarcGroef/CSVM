#include <csvm/csvm_clean_descriptor.h>

using namespace std;
using namespace csvm;


//simply copy all pixels in grey-value towards a feature vector
Feature CleanDescriptor::describe(Patch p){  
   
   unsigned int imHeight = p.getHeight();
   unsigned int imWidth = p.getWidth();
   unsigned int imSize = (imWidth * imHeight * 3);
 
   
   Feature f(imSize,0);
   
   
   f.content = vector<double>(imSize,0);
   
   /*for(size_t idxX = 0; idxX < imWidth; ++idxX){
      for(size_t idxY = 0; idxY < imHeight; ++idxY){
         f.content[idxY * imWidth + idxX] = (double)(p.getGreyPixel(idxX,idxY)) ;
      }
   }*/
   
   for(size_t chIdx = 0; chIdx < 3; ++chIdx){
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            f.content[idxY * imWidth + idxX] = (double)(p.getPixel(idxX,idxY,chIdx)) ;
         }
      }
   }
   return f;
   
}  