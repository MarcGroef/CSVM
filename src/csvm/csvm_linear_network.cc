#include <csvm/csvm_linear_network.h>
#include <cmath>
using namespace std;
using namespace csvm;

LinNetwork::LinNetwork(){//(unsigned int nClasses, unsigned int nCentroids, double initWeights){
   //nClasses = 10;
   //nCentroids = 200 * 4;
   //weights.reserve(nClasses);
   
   
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

void LinNetwork::setSettings(LinNetSettings s){
   settings = s;
   
   settings.nClasses = 10;
  settings.nCentroids = 1000;
   biases = vector<double>(settings.nClasses,0);
   
   weights = vector< vector<double> > (settings.nClasses, vector<double>(settings.nCentroids * 4,settings.initWeight));
   biases = vector<double>(settings.nClasses,0);
   
    //cout << "linnet settins set: nCentroids = " << settings.nCentroids << "learnRate = " << settings.learningRate <<  endl;
   
  
}

double sigmoid(double x){
   return 1.0/(1.0 + exp(-1.0 * x));
}
double LinNetwork::computeOutput(unsigned int networkClassIdx,vector<double>& clActivations){
   double out = 0.0;

   for(size_t centrIdx = 0; centrIdx < settings.nCentroids * 4; ++centrIdx){
      //cout << "weights = " <<  weights[networkClassIdx][clIdx][centrIdx] << ", activ = " << clActivations[clIdx][centrIdx] << ", sum = " << out <<endl;
      out += weights[networkClassIdx][centrIdx] * clActivations[centrIdx];
   }
 
   //cout << "entering sigmoid: out = " << out << ", bias = " << biases[networkClassIdx] << endl;
   //out = sigmoid(out + biases[networkClassIdx]);
   //cout << out << endl;
   out += biases[networkClassIdx];
   return out;
}

void LinNetwork::train(vector< vector< double > >& activations, CSVMDataset* ds){
   unsigned int nIter = 5000;
   unsigned int nData = activations.size();
   double output;
   double error;
   double target;
   double learningRate = settings.learningRate;
   double deltaWeight;
   double errorSum = 100;
   double sqErrorSum = 100;
   double sumOfChange;
   
   cout << "Entering linnet training: LearningRate = " << learningRate << ", nCentroids = " << settings.nCentroids << endl;
   for(size_t networkClassIdx = 0; networkClassIdx < settings.nClasses; ++networkClassIdx){
      sumOfChange = 1000.0;
      errorSum = 100;
      sqErrorSum = 1000000000000;
      cout << "beginning training round\n";
      
      for(size_t iterIdx = 0; iterIdx < nIter /*&& 0.5 * sqErrorSum / nData > 0.04*/ /*sumOfChange > 0.0001*/; ++iterIdx){
         errorSum = 0.0;
         unsigned int nTrained = 0;
         sqErrorSum = 0.0;
         sumOfChange = 0.0;
         for(size_t dataIdx = 0; dataIdx < nData; ++ dataIdx){
            
            target = networkClassIdx == ds->getImagePtr(dataIdx)->getLabelId() ? 10.0 : -10.0;
            output = computeOutput(networkClassIdx, activations[dataIdx]);
            
            error = target - (output);
            sqErrorSum += error * error;
            errorSum += error;
            ++nTrained;
            
            for(size_t centrIdx = 0; centrIdx < settings.nCentroids * 4; ++centrIdx){
               
               double activ = activations[dataIdx][centrIdx];
               deltaWeight = learningRate * error * -1.0 /* output * (1.00 - output) */* activ;
               sumOfChange += (deltaWeight < 0 ? deltaWeight * -1 : deltaWeight);
               
               weights[networkClassIdx][centrIdx] -= deltaWeight;
            }
            
            //sumOfChange += deltaBias < 0 ? -1.0 * deltaBias : deltaBias;
            
            
           // biases[networkClassIdx] -= deltaBias;
         }
         if(iterIdx%100==0)
            cout << "Network " << networkClassIdx << " SOC = " << sumOfChange << "   \terror = " <<  0.5 * sqErrorSum /nData <<endl;
         
         biases[networkClassIdx] -= learningRate*  -1*errorSum / nData;
      }
      
      
      
   }
}

unsigned int LinNetwork::classify(vector< double > imageActivations){
   unsigned int maxLabel = 0;
   double maxOutput = computeOutput(0, imageActivations);
   double output;
   //cout << "Classify!\n";
   //cout << "output 0 = " << maxOutput << endl;
   for(size_t labelIdx = 1; labelIdx < settings.nClasses; ++labelIdx){
      output = computeOutput(labelIdx, imageActivations) ;
      //cout << "output " << labelIdx <<" = " << output << endl;
      if(output > maxOutput){
         maxOutput = output;
         maxLabel = labelIdx;
      }
   }
   return maxLabel;
}
