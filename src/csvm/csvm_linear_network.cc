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
   //cout << "linset learnRate = " << settings.learningRate << endl;
   bool oneCl = !settings.useDifferentCodebooksPerClass;
   
   settings.nClasses = 10;
  settings.nCentroids = 1000;
   biases = vector<double>(settings.nClasses,0);
   
   weights = vector< vector< vector<double> > >(settings.nClasses, vector< vector< double> >(oneCl ? 1 : settings.nClasses, vector<double>(settings.nCentroids * 4,settings.initWeight)));
   biases = vector<double>(settings.nClasses,0);
   
    //cout << "linnet settins set: nCentroids = " << settings.nCentroids << "learnRate = " << settings.learningRate <<  endl;
   
  
}

double sigmoid(double x){
   return 1.0/(1.0 + exp(-1.0 * x));
}
double LinNetwork::computeOutput(unsigned int networkClassIdx, vector< vector<double> >& clActivations){
   double out = 0.0;
   //for all codebooks from different classes
   bool oneCl = !settings.useDifferentCodebooksPerClass;
   for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < settings.nClasses; ++clIdx){
      for(size_t centrIdx = 0; centrIdx < settings.nCentroids * 4; ++centrIdx){
        //cout << "weights = " <<  weights[networkClassIdx][clIdx][centrIdx] << ", activ = " << clActivations[clIdx][centrIdx] << ", sum = " << out <<endl;
         out += weights[networkClassIdx][clIdx][centrIdx] * clActivations[clIdx][centrIdx];
      }
   }//cout << out << endl;
   //cout << "entering sigmoid: out = " << out << ", bias = " << biases[networkClassIdx] << endl;
   //out = sigmoid(out + biases[networkClassIdx]);
   //cout << out << endl;
   //out += biases[networkClassIdx];
   return out;
}

void LinNetwork::train(vector< vector< vector< double > > >& activations, CSVMDataset* ds){
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
   bool oneCl = !settings.useDifferentCodebooksPerClass;
   cout << "Entering linnet training: LearningRate = " << learningRate << ", nCentroids = " << settings.nCentroids << endl;
   for(size_t networkClassIdx = 0; networkClassIdx < settings.nClasses; ++networkClassIdx){
      sumOfChange = 1000.0;
      errorSum = 100;
      sqErrorSum = 1000000000000;
      cout << "beginning training round\n";
      //for(size_t networkClassIdx = 0; networkClassIdx < nClasses; ++networkClassIdx){
      for(size_t iterIdx = 0; iterIdx < nIter /*&& 0.5 * sqErrorSum / nData > 0.04*/ /*sumOfChange > 0.0001*/; ++iterIdx){
         errorSum = 0.0;
         unsigned int nTrained = 0;
         sqErrorSum = 0.0;
         sumOfChange = 0.0;
         for(size_t dataIdx = 0; dataIdx < nData; ++ dataIdx){
            
            target = networkClassIdx == ds->getImagePtr(dataIdx)->getLabelId() ? 10.0 : -10.0;
            output = computeOutput(networkClassIdx, activations[dataIdx]);
            //cout << "trainoutput: " << output << endl;
            error = target - (output);
            /*error = error > 100 ? 100 : error;
            error = error < -100 ? -100 : error;*/
            //cout << output << endl;
            sqErrorSum += error * error;
            errorSum += error;
            //if(error > 0){ //update weight iff not already correct output
               ++nTrained;
               for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < settings.nClasses; ++clIdx){
                  for(size_t centrIdx = 0; centrIdx < settings.nCentroids * 4; ++centrIdx){
                     
                     //cout << "activations = " << activations[dataIdx][clIdx][centrIdx] << endl;
                     double activ = activations[dataIdx][clIdx][centrIdx];//> 0 ? activations[dataIdx][clIdx][centrIdx] : 0;
                     deltaWeight = learningRate * error * -1.0 /* output * (1.00 - output) */* activ;
                     //cout << "centr " << centrIdx <<  " weight = " << weights[networkClassIdx][clIdx][centrIdx] << ", newWeight = " << (weights[networkClassIdx][clIdx][centrIdx]-deltaWeight) << " deltaWeight = " << deltaWeight << ", out = " << output << ", error = " << error << ", act = " << activ << endl;
                     
                     sumOfChange += (deltaWeight < 0 ? deltaWeight * -1 : deltaWeight);
                     
                     weights[networkClassIdx][clIdx][centrIdx] -= deltaWeight;
                  }
               }
            //}   
               //sumOfChange += deltaBias < 0 ? -1.0 * deltaBias : deltaBias;
            
            
           // biases[networkClassIdx] -= deltaBias;
         }
         if(iterIdx%100==0)
            cout << "Network " << networkClassIdx << " SOC = " << sumOfChange << "   \terror = " <<  0.5 * sqErrorSum /nData <<endl;
         //else cout << "this data should be classified correctly\n";
         double deltaBias =  learningRate * -1.0 /*  (output) * (1.0 - output) */*  error;
         biases[networkClassIdx] -= learningRate*  -1*errorSum / nData;
      }
      
      
      
   }
}

unsigned int LinNetwork::classify(vector< vector< double > > imageActivations){
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
