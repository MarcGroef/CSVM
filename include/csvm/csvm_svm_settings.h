#ifndef CSVM_SVM_SETTINGS_H
#define CSVM_SVM_SETTINGS_H

using namespace std;
namespace csvm{
  
    class SVMSettings{
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
      
    };
  
}

#endif