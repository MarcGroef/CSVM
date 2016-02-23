#include <csvm/csvm_whitening.h>




void Whitener::calculateMatrices(vector<Feature>& collection){
   size_t collectionSize = collection.size();
   
   //substract mean from data.
   size_t nDims;
   
   for(size_t xIdx = 0; xIdx != collectionSize; ++xIdx){
      double mean = 0;
      nDims = collection[0].content.size();
      
      for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
         mean += collection[xIdx].content[dIdx];
      }
      mean /= nDims;
      
      for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
        collection[xIdx].content[dIdx] -= mean;
      }
   }
   
   //calc sigma matrix;
   sigma.resize(nDims, vector<double>(nDims, 0));
   
   for(size_t xIdx = 0; xIdx != collectionSize; ++xIdx){
      for(size_t d1Idx = 0; d1Idx != nDims; ++d1Idx){
         for(size_t d2Idx = 0; d2Idx != nDims; ++d2Idx){
            sigma[d1Idx][d2Idx] += collection[xIdx].content[d1Idx] * collection[xIdx].content[d1Idx];
         }
      }
   }
   
   for(size_t d1Idx = 0; d1Idx != nDims; ++d1Idx){
      for(size_t d2Idx = 0; d2Idx != nDims; ++d2Idx){
         sigma[d1Idx][d2Idx] /= collectionSize;
      }
   }
   
   //now get the eigenvectors of this thing.
   
   
   
   
}