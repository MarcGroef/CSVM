#include<csvm/csvm_conv_svm.h>

using namespace std;
using namespace csvm;

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      
      cout << "nCentroids = " << settings.nCentroids << endl;
      cout << "nClasses = " << settings.nClasses << endl;
      cout << "initWeight= " << settings.initWeight << endl;
      weights = vector< vector<double> >(settings.nClasses, vector<double>(settings.nCentroids, settings.initWeight));
      biases = vector<double>(settings.nClasses, 0);
      dSlacks = vector<double>(0);

   }
   
   double ConvSVM::output(vector< vector<double> >& activations, unsigned int svmIdx){
      unsigned int nCentroids = activations[0].size();
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
         //cout << "weights = " << weights[svmIdx][centrIdx] << endl;
         out += weights[svmIdx][centrIdx] * activations[0][centrIdx];
      }
      out += biases[svmIdx];
      return out;
   }
   
   void ConvSVM::train(vector< vector< vector<double> > >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      if(dSlacks.size() == 0)
         dSlacks = vector<double>(nData, 0);
      //for all csvms in ensemble
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         double tempBias = biases[svmIdx];
         for(size_t dIdx = 0; dIdx < nData; ++dIdx){
            dSlacks[dIdx] = 0;
            unsigned int label = ds->getImagePtr(dIdx)->getLabelId();
            double yData = (label == svmIdx ? 1.0 : -1.0);
            double out = output(activations[dIdx], svmIdx);
            if(yData * out < 1){
               
               for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                  dSlacks[dIdx] += yData * activations[dIdx][0][clIdx];
               }
            }
         }
         unsigned int nError = 0;
         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            double sumSlack = 0;
            
            //double tempBias = biases[svmIdx];//0;
            nError = 0;
            //for all data
            //cout << "sumdSlack = " << sumDSlack << endl;
            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               tempBias = 0;
               double sumDSlack = 0;
               unsigned int label = ds->getImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               
               double out = output(activations[dIdx], svmIdx);
               
               
               //update dSlack for this data
               dSlacks[dIdx] = 0;
               if(yData * out < 1){
                  
                  for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                     dSlacks[dIdx] += yData * activations[dIdx][0][clIdx];
                  }
               }
               
               //calculate new dSlack sum
               for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
                  sumDSlack += dSlacks[dIdx1];
               }
               
               
               
               
               double weightSum = 0;
               sumSlack += yData * out < 0 ? yData * out * -1 : yData * out;
               //determine slack differential
               
               if(yData * out < 1){  //otherwise keep it zero
                  ++nError;
                  //update weights
                  for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                     //cout << "weight = " << weights[svmIdx][clIdx]  << "\tsumDSlack = " << sumDSlack << "\tbias = " << biases[svmIdx] << endl;
                     //weights[svmIdx][clIdx] -= settings.learningRate * ( weights[svmIdx][clIdx] - settings.CSVM_C * yData * out) ;
                     weights[svmIdx][clIdx] -= settings.learningRate * ( weights[svmIdx][clIdx] - settings.CSVM_C * sumDSlack) ;
                     
                  }
                  tempBias += /*settings.learningRate */ out * yData; 
               }
                           
               
            }
            tempBias /= nError;
            /// nError;
            double objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack;
            biases[svmIdx] = tempBias ;
            if(itIdx % 100 == 0)cout << "CSVM " << svmIdx << ": Objective = " << objective << ", sumSlack = " << sumSlack << endl;   
         }
         //biases[svmIdx] = tempBias ;
      }
   }
   
   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector< vector<double> >& activations){
      unsigned int maxLabel = 0;
      double maxOut = output(activations, 0);
      //cout << "out 0 = " << maxOut << endl;
      //cout << "settings.nClasses = " << settings.nClasses << endl;
      for(size_t svmIdx = 1; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         //cout << "out " << svmIdx << " = " << out << " with bias = " << biases[svmIdx] <<endl;
         if(out > maxOut){
            maxOut = out;
            maxLabel = svmIdx;
            //cout << "Maxlabel = " << maxLabel << endl;
         }
      }
      return maxLabel;
      
   }
