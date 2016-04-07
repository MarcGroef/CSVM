#ifndef CSVM_WHITENING_H
#define CSVM_WHITENING_H

#include <vector>
#include "csvm_feature.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

namespace csvm{
   
   class Whitener{
      //vector< vector<double> > sigma;
      MatrixXd sigma;
      MatrixXcd eigenVectors;
      MatrixXd pc;
   public:
      void analyze(vector<Feature>& collection);
      void transform(Feature& f);
   };
   
   
}

#endif