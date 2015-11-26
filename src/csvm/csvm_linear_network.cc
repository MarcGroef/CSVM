#include <csvm/csvm_linear_network.h>

using namespace std;
using namespace csvm;

LinNetwork::LinNetwork(){//(unsigned int nClasses, unsigned int nCentroids, double initWeights){
   nClasses = 10;
   nCentroids = 1000 * 4;
   //weights.reserve(nClasses);
   weights = vector< vector< vector<double> > >(nClasses, vector< vector< double> >(1, vector<double>(nCentroids * 4,.0001)));
   biases = vector<double>(nClasses,0);
   //for(size_t networkClassIdx = 0; networkClassIdx < nClasses; ++networkClassIdx){
      //weights[networkClassIdx] = vector< vector<double> >(nClasses);
      /*for(size_t clIdx = 0; clIdx < nClasses; ++ clIdx){
         //weights[networkClassIdx][clIdx] = vector<double>(nCentroids);
         for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
            weights[networkClassIdx][clIdx][centrIdx] = initWeights;
         }
      }*/
  // }
}

double sigmoid(double x){
   return 1.0/(1.0 - exp(-1.0 * x));
}
double LinNetwork::computeOutput(unsigned int networkClassIdx, vector< vector<double> >& clActivations){
   double out = 0.0;
   //for all codebooks from different classes
   for(size_t clIdx = 0; clIdx < 1 && clIdx < nClasses; ++clIdx){
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
        //cout << "weights = " <<  weights[networkClassIdx][clIdx][centrIdx] << endl;
         out += weights[networkClassIdx][clIdx][centrIdx] * clActivations[clIdx][centrIdx];
      }
   }//cout << out << endl;
   //out = sigmoid(out);
   //cout << out << endl;
   out += biases[networkClassIdx];
   return out;
}

void LinNetwork::train(vector< vector< vector< double > > >& activations, CSVMDataset* ds){
   unsigned int nIter = 500;
   unsigned int nData = activations.size();
   double output;
   double error;
   double target;
   double learningRate = 0.02;
   double deltaWeight;
   double errorSum = 100;
   double sqErrorSum = 100;
   double sumOfChange;
   for(size_t networkClassIdx = 0; networkClassIdx < nClasses; ++networkClassIdx){
      sumOfChange = 1000.0;
      errorSum = 100;
      sqErrorSum = 100;
      cout << "beginning training round\n";
      //for(size_t networkClassIdx = 0; networkClassIdx < nClasses; ++networkClassIdx){
      for(size_t iterIdx = 0; /*iterIdx < nIter &&*/ 0.5 * sqErrorSum > 0.000001 /*sumOfChange > 0.1*/; ++iterIdx){
         errorSum = 0.0;
         sumOfChange = 0.0;
         for(size_t dataIdx = 0; dataIdx < nData; ++ dataIdx){
            target = networkClassIdx == ds->getImagePtr(dataIdx)->getLabelId() ? 1.0 : 0.0;
            output = computeOutput(networkClassIdx, activations[dataIdx]);
            //cout << "output: " << output << endl;
            error = target - (output);
            /*error = error > 100 ? 100 : error;
            error = error < -100 ? -100 : error;*/
            //cout << output << endl;
            sqErrorSum += error * error;
            errorSum += error;
            //if(error > 0){ //update weight iff not already correct output
               for(size_t clIdx = 0;clIdx < 1 && clIdx < nClasses; ++clIdx){
                  for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
                     deltaWeight = learningRate * error * -1 */* output * (1.0 - output) */ activations[dataIdx][clIdx][centrIdx];
                     //cout << "activations = " << activations[dataIdx][clIdx][centrIdx] << endl;
                     cout << "weight = " << weights[networkClassIdx][clIdx][centrIdx] << " deltaWeight = " << deltaWeight << ", error = " << error << ", act = " << activations[dataIdx][clIdx][centrIdx] << endl;
                     sumOfChange += (deltaWeight < 0 ? deltaWeight * -1 : deltaWeight);
                     
                     weights[networkClassIdx][clIdx][centrIdx] -= deltaWeight;
                  }
               }
               //double deltaBias =  learningRate * -1 *  output * (1.0 - output) *  error;
               //biases[networkClassIdx] -= deltaBias;
               //sumOfChange += deltaBias < 0 ? -1.0 * deltaBias : deltaBias;
            //}
         }
         /*if(iterIdx%100==0)*/
         biases[networkClassIdx] =  errorSum / nData;
         cout << "Network " << networkClassIdx << " SOC = " << sumOfChange << "\terror = " <<  0.5 * sqErrorSum <<endl;
         //else cout << "this data should be classified correctly\n";
      }
      
      
      
   }
}

unsigned int LinNetwork::classify(vector< vector< double > > imageActivations){
   unsigned int maxLabel = 0;
   unsigned int maxOutput = computeOutput(0, imageActivations);
   double output;
   for(size_t labelIdx = 1; labelIdx < nClasses; ++labelIdx){
      output = computeOutput(labelIdx, imageActivations) ;
      if(output > maxOutput){
         maxOutput = output;
         maxLabel = labelIdx;
      }
   }
   return maxLabel;
}