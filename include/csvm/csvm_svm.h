#ifndef CSVM_SVM_H
#define CSVM_SVM_H

#include "csvm_feature.h"


using namespace std;
namespace csvm{
  
  
  
  struct SVM_Settings{
    double alpha; 
    double beta;
    double COST;
    double D2;
    double SVM_C;
    double EPS;
    double SIGMA;
    double INIT_ALPHA;
    int ALPHA_ITER;
    int NR_REP1;
    int NR_REP2;
  
  }
   
  class SVM{
      
    //State variables
    struct SVM_Settings;
    
    string parameterFile;
    string train
      
  public:
     
     
    void setSettings(double alpha, double beta, double COST, double D2, double SVM_C, double EPS,
      double SIGMA, double INIT_ALPHA, int ALPHA_ITER, int NR_REP1, int NR_REP2);
    
    double classify(Feature feat);  //classify the feature and return output
    void train(Feature feat);
     
      
  }
   
}

#endif