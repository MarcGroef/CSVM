#include <csvm/csvm_clean_descriptor.h>

using namespace std;
using namespace csvm;

Feature CleanDescriptor::describe(Patch p){  
   
   unsigned int imHeight = p.getHeight();
   unsigned int imWidth = p.getWidth();
   unsigned int imSize = (imWidth * imHeight);
 
   
   Feature f(imSize,0);
   
   f.content.reserve(imSize);
   
   
   for(size_t idxX = 0; idxX < imWidth; ++idxX){
      for(size_t idxY = 0; idxY < imHeight; ++idxY){
         f.content[idxY * imWidth + idxX] = (double)(p.getGreyPixel(idxX,idxY));
      }
   }
   
   return f;
   
}  