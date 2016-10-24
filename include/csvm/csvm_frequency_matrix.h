#ifndef CSVM_FREQUENCY_MATRIX_H
#define CSVM_FREQUENCY_MATRIX_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
   using namespace std;
   
   namespace csvm{
      
      
      class FreqMatrix{
         vector<float> triangle;
         int size;
         float mean;
         float stddev;
	 float totalInput;
      public:
        FreqMatrix(int size) ;
	FreqMatrix();
        ~FreqMatrix();
	void reserve(int size);
        int getTriangleIndex(int i, int j);
        void setZeros();
        void addCombo(int wordA, int wordB, float activationA, float activationB);
        float getCombo(int wordA, int wordB);
        void analyze();
	void normalize();
      };
      
      
      
   }

#endif