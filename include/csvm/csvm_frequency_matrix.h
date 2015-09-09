#ifndef CSVM_FREQUENCY_MATRIX_H
#define CSVM_FREQUENCY_MATRIX_H

#include <iostream>
#include <cstdlib>
#include <cmath>

   using namespace std;
   
   namespace csvm{
      
      
      class FreqMatrix{
         float* triangle;
         int size;
         float mean;
         float stddev;
      public:
        FreqMatrix(int size) ;
        ~FreqMatrix();
        int getTriangleIndex(int i, int j);
        void setZeros();
        void addCombo(int wordA, int wordB, float activationA, float activationB);
        float getCombo(int wordA, int wordB);
        void analyze();
      };
      
      
      
   }

#endif