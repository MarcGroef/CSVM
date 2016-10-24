#include <csvm/csvm_frequency_matrix.h>
//DEPRECATED
using namespace std;
using namespace csvm;

FreqMatrix::FreqMatrix(){
  
}

FreqMatrix::FreqMatrix(int nWords){
   reserve(nWords);
}

FreqMatrix::~FreqMatrix(){

}

void FreqMatrix::reserve(int nWords){
  size = (int)(0.5 * (nWords * nWords + nWords));
  triangle.reserve(size);
  totalInput = 0;
  setZeros();
}

int FreqMatrix::getTriangleIndex(int i, int j){
   if(i > size - 1 || j > size - 1){
      cout << "csvm_frequency_matrix::FreqMatrix::getTriangleIndex(int,int) Error: Provided index (" << i << "," << j << ")out of bounds.\nExiting..\n";
      exit(-1);
   }
   i++;
   j++;
   return (int)( 0.5 * (j * j - j) + i - 1);
}

void FreqMatrix::setZeros(){
   for(int i = 0; i < size; i++)
      triangle[i]=0;
 
}

void FreqMatrix::addCombo(int wordA, int wordB, float activationA, float activationB){
   triangle[getTriangleIndex(wordA, wordB)] += activationA * activationB;
   totalInput += activationA * activationB;
}

void FreqMatrix::normalize(){
   for(int i = 0; i < size; i++){
      triangle[i] /= totalInput;
   }
}

float FreqMatrix::getCombo(int wordA, int wordB){
   return triangle[getTriangleIndex(wordA, wordB)];
}

void FreqMatrix::analyze(){
   mean=0;
   stddev=0;
   for(int i = 0; i < size; i++){
      mean += triangle[i];
   }
   mean /= size;
   for(int i = 0; i < size; i++){
      stddev += (mean - triangle[i])*(mean - triangle[i]);
   }
   stddev = sqrt(stddev);
   stddev /= size;
}