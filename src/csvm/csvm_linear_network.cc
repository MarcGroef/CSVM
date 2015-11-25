#include <csvm/csvm_linear_network.h>

using namespace std;
using namespace csvm;

LinNetwork::LinNetwork(unsigned int nClasses, unsigned int nCentroids, double initWeights){
   this->nClasses = nClasses;
   this->nCentroids = nCentroids;
   weights.reserve(nClasses);
   for(size_t clIdx = 0; clIdx < nClasses; ++ clIdx){
      weights[clIdx].reserve(nCentroids);
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
         weights[clIdx][centrIdx] = initWeights;
      }
   }
}

double LinNetwork::computeOutput(unsigned int networkClassIdx, vector< vector<double> >& clActivations){
   double out = 0.0;
   //for all codebooks from different classes
   for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         out += weights[networkClassIdx][clIdx][centrIdx] * clActivations[clIdx][centrIdx];
      }
   }
   return out;
}

void LinNetwork::train(vector< vector< vector< double > > >& clActivations, CSVMDataset* ds){
   unsigned int nIter = 100;
   unsigned int nData = 1000;
   double output;
   double error;
   double target;
   double learningRate = 0.05;
   double deltaWeight;
   double sumOfChange;
   for(size_t iterIdx = 0; iterIdx < nIter; ++iterIdx){
      sumOfChange = 0.0
      for(size_t networkClassIdx = 0; networkClassIdx < nClasses; ++networkClassIdx){
         for(size_t dataIdx = 0; dataIdx < nData; ++ dataIdx){
            target = networkClassIdx == ds->getImagePtr(dataIdx) ? 1.0 : -1.0;
            output = computeOutput(networkClassIdx, clActivations[iterIdx]);
            error = target - output;
            if(error > 0){ //update weight iff not already correct output
               for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
                  for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
                     deltaWeight = learningRate * error * clActivations[dataIdx][clIdx][centrIdx];
                     sumOfChange += deltaWeight;
                     weights[networkClassIdx][clIdx][centrIdx] -= deltaWeight;
                  }
               }
            }
         }
      }
      cout << "SOC = " << sumOfChange << endl;
   }
}

unsigned int LinNetwork::classify(< vector< vector< double > >& imageActivations)[
   unsigned int maxLabel = 0;
   unsigned int maxOutput = computeOutput(0, imageActivations);
   double output;
   for(labelIdx = 1; labelIdx < nClasses; ++labelIdx){
      output = computeOutput(labelIdx);
      if(output > maxOutput){
         maxOutput = output;
         maxLabel = labelIdx;
      }
   }
   return maxLabel;
}