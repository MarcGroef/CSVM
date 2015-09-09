#ifndef CSVM_FREQUENCY_MATRIX_H
#define CSVM_FREQUENCY_MATRIX_H

#include <iostream>
#include <cstdlib>
#include <cmath>

   using namespace std;
   
   namespace csvm{
      
      
      class FreqMatrix{
         double* triangle;
         int size;
         double mean;
         double stddev;
      public:
        FreqMatrix(int size) ;
        ~FreqMatrix();
        int getTriangleIndex(int i, int j);
        void setZeros();
        void addCombo(int wordA, int wordB, double activationA, double activationB);
        double getCombo(int wordA, int wordB);
        void analyze();
      };
      
      
      
   }

#endif