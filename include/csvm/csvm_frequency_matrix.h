#ifndef CSVM_FREQUENCY_MATRIX_H
#define CSVM_FREQUENCY_MATRIX_H

#include <iostream>
#include <cstdlib>

   using namespace std;
   
   namespace csvm{
      
      
      class FreqMatrix{
         float* triangle;
         int size;
         
      public:
        FreqMatrix(int size) ;
        ~FreqMatrix();
        int getTriangleIndex(int i,int j);
        void setZeros();
        void addCombo(int wordA,int wordB,float activationA,float activationB);
      };
      
      
      
   }

#endif