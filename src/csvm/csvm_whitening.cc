#include <csvm/csvm_whitening.h>

using namespace std;
using namespace csvm;
using namespace Eigen;


void Whitener::analyze(vector<Feature>& collection){
   cout << "analyzing features....\n";
   //return;
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
   MatrixXd eigenvalues = es.eigenvalues().real();
   cout << "eigenVectors: nRows: "  << eigenVectors.rows() << ", nCols = " << eigenVectors.cols() << endl;// * (eigenVectors.real().adjoint());
   cout << "eigenvalues: nRows: "  << eigenvalues.rows() << ", nCols = " << eigenvalues.cols() << endl;// * (eigenVectors.real().adjoint());
   cout << "I have eigen vectors me\n";
   //return;
   MatrixXd t = MatrixXd((eigenvalues.array() + 0.1).cwiseInverse().sqrt()).asDiagonal();
   cout << "T: nRows: "  << t.rows() << ", nCols = " << t.cols() <<endl;// * (eigenVectors.real().adjoint());
   pc = (eigenVectors.real() * t ) * (eigenVectors.real().adjoint());
   //cout << "nRows: "  << MatrixXd(eigenVectors.real() * t).rows() << ", nCols = " << MatrixXd(eigenVectors.real() * t).cols() <<endl;// * (eigenVectors.real().adjoint());
   //cout << "nRows: "  << (eigenVectors.real().adjoint()).rows() << ", nCols = " << (eigenVectors.real().adjoint()).cols() <<endl;//
   cout << "Done with PCA!\n";
   cout << "nRows: "  << pc.rows() << ", nCols = " << pc.cols() << endl;//
}

void Whitener::transform(Feature& f){
   //return;
   size_t dims = f.content.size();
   //cout << "trasnform!\n";
   double mean = 0;
   
   for(size_t dIdx = 0; dIdx != dims; ++dIdx){
      mean += f.content[dIdx];
   }
   mean /= dims;
   for(size_t dIdx = 0; dIdx != dims; ++dIdx){
      f.content[dIdx] -= mean;
   }
   
   
   //subtracted mean, now map to PCA space
   double* data = f.content.data();
   VectorXd m = Map<VectorXd>(data, dims);
   
   //cout << m << endl;
   //return;
   VectorXd res = m.transpose() * pc;
   

   f.content = vector<double>(res.data(), res.data() + res.rows() * res.cols());
   if(f.content.size() != dims)
      cout << "WARNING: Whitener::transform() I/O dims dont match!\n";

}