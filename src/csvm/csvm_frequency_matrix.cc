#include <csvm/csvm_frequency_matrix.h>

using namespace std;
using namespace csvm;

FreqMatrix::FreqMatrix(int nWords){
   size = 0.5 * (nWords * nWords + nWords);
   triangle = new float[size]();
}

FreqMatrix::~FreqMatrix(){
   delete triangle;  
}


int FreqMatrix::getTriangleIndex(int i,int j){
   if(i > size - 1 || j > size - 1){
      cout << "csvm_frequency_matrix::FreqMatrix::getTriangleIndex(int,int) Error: Provided index (" << i << "," << j << ")out of bounds.\nExiting..\n";
      exit(-1);
   }
   i++;
   j++;
   return 0.5 * (j * j - j) + i - 1;
}

void FreqMatrix::setZeros(){
   for(int i = 0; i < size; i++)
      triangle[i]=0;
 
}

void FreqMatrix::addCombo(int wordA, int wordB, float activationA, float activationB){
   triangle[getTriangleIndex(wordA, wordB)] += activationA * activationB;
}

float FreqMatrix::getCombo(int wordA, int wordB){
   return triangle[getTriangleIndex(wordA, wordB)];
}