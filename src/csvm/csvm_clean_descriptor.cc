#include <csvm/csvm_clean_descriptor.h>
#include <cmath>
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
   double mean = 0.0;
   double stddev = 0.0;
   for(size_t chIdx = 0; chIdx < 3; ++chIdx){
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            f.content[idxY * imWidth + idxX] = (double)(p.getPixel(idxX,idxY,chIdx)) ;
            mean += f.content[idxY * imWidth + idxX];
            
         }
      }
   }
   mean /= (3 * imWidth * imHeight);
   for(size_t chIdx = 0; chIdx < 3; ++chIdx){
      for(size_t idxX = 0; idxX < imWidth; ++idxX){
         for(size_t idxY = 0; idxY < imHeight; ++idxY){
            
            stddev += (mean - f.content[idxY * imWidth + idxX]) * (mean - f.content[idxY * imWidth + idxX]);
            
         }
      }
   }
   //cout << "var = " << stddev << ", nVals = " << (3 * imWidth * imHeight) << endl;
   stddev /= (3 * imWidth * imHeight);
   stddev = sqrt(stddev);
   //cout << "stddev = " << stddev << endl;
   if (stddev > 0){
      for(size_t chIdx = 0; chIdx < 3; ++chIdx){
         for(size_t idxX = 0; idxX < imWidth; ++idxX){
            for(size_t idxY = 0; idxY < imHeight; ++idxY){
               f.content[idxY * imWidth + idxX] = (f.content[idxY * imWidth + idxX] - mean)/stddev;
               //cout << "newVal = " << f.content[idxY * imWidth + idxX] << endl;
            }
         }
      }
   }
   return f;
   
}  