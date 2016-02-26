#include <csvm/csvm_whitening.h>

using namespace std;
using namespace csvm;
using namespace Eigen;


void Whitener::analyze(vector<Feature>& collection){
   size_t collectionSize = collection.size();
   
   //substract mean from data.
   size_t nDims = collection[0].content.size();
   
   for(size_t xIdx = 0; xIdx != collectionSize; ++xIdx){
      double mean = 0;
      
      
      for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
         mean += collection[xIdx].content[dIdx];
      }
      mean /= nDims;
      
      for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
        collection[xIdx].content[dIdx] -= mean;
      }
   }
   
   //calc sigma matrix;
   //sigma.resize(nDims, vector<double>(nDims, 0));
   sigma = MatrixXd::Constant(nDims, nDims,0);
   
   for(size_t xIdx = 0; xIdx != collectionSize; ++xIdx){
      for(size_t d1Idx = 0; d1Idx != nDims; ++d1Idx){
         for(size_t d2Idx = 0; d2Idx != nDims; ++d2Idx){
            sigma(d1Idx,d2Idx) += collection[xIdx].content[d1Idx] * collection[xIdx].content[d2Idx];
         }
      }
   }
   
   sigma /= collectionSize;

   
   //now get the eigenvectors of this thing.
   EigenSolver<MatrixXd> es(sigma, true);
   eigenVectors = es.eigenvectors();
   
   
   VectorXd t = MatrixXd((MatrixXd(es.eigenvalues().real().asDiagonal()).array() + 0.1).cwiseInverse().array().sqrt()).diagonal();
   pc = eigenVectors.real() * t;// * eigenVectors.real().adjoint();
   cout << "Done with PCA!\n";
}

void Whitener::transform(Feature& f){
   size_t dims = f.content.size();
   
   for(size_t dIdx = 0; dIdx != dims; ++dIdx){
      f.content[dIdx] -= means[dIdx];
   }
   
   
   //subtracted mean, now map to PCA space
   double* data = f.content.data();
   VectorXd m = Map<VectorXd>(data, dims);
   
   VectorXd res = m * pc;
   

   f.content = vector<double>(res.data(), res.data() + res.rows() * res.cols());


}