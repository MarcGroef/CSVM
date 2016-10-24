#include<csvm/csvm_conv_svm.h>
/* Patch-based SVM interpretation. 
 * Concept from M.A. Wiering (2015/2016)
 * 
 * NEEDS CLEANUP FOR RELEASE
 * 
 * 
 */
using namespace std;
using namespace csvm;

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      
   }
	
	

   // classification given an activation vector
   float ConvSVM::output(vector<float>& activations, unsigned int svmIdx){
      
      unsigned int nCentroids = activations.size();
      float out = 0;
      
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         out += weights[svmIdx][centrIdx] * activations[centrIdx];
      }
      
      out += biases[svmIdx];
      return out;
   }
   


   // training
   void ConvSVM::train(vector< vector<float> >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      settings.nCentroids = activations[0].size();
      cout << fixed << setprecision(5);
      
      weights = vector< vector<float> >(settings.nClasses, vector<float>(settings.nCentroids, settings.initWeight));
      biases  = vector<float>(settings.nClasses, 0);

      //############ Logging functions ################
      maxOuts = vector<float>(settings.nClasses, 0);
      minOuts = vector<float>(settings.nClasses, 0);
      avOuts  = vector<float>(settings.nClasses, 0);
      ofstream statDatFile;
      //###############################################

      //for all csvms in ensemble
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         //############ Logging functions ################
         stringstream ss;
         ss << "statData_SVM-" << svmIdx << ".csv";
         string fName = ss.str();
         statDatFile.open ( fName.c_str() );
         statDatFile << "Iteration,Objective,Score,MinOut,MaxOut,stdDevMinOutPos,stdDevMinOutNeg,StdDevMaxOutPos,StdDevMaxOutNeg,HyperplanePercentage" << endl;
         if (debugOut) cout << "\n\nSVM " << svmIdx << ":\t\t\t(data written to " << fName << ")\n" << endl;
         //###############################################
         
         //for all training iterations
         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            
            float sumSlack = 0;
            float right = 0;
            float wrong = 0;

            //############ Logging functions ################
            allOuts  = vector<float>(nData, 0);
            minOut = 0;
            maxOut = 0;
            nMax   = 0;
            nMin   = 0;
            //###############################################
         

            // for all datapoints
            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               unsigned int label = ds->getTrainImagePtr(dIdx)->getLabelId();
               float yData = (label == svmIdx ? 1.0 : -1.0);
               float out = output(activations[dIdx], svmIdx);
             
               // for all centers 
               for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){

                  // partial derivatives to the weights
                  if(yData * out < 1){
                     if (not settings.L2){
                        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) -  yData * activations[dIdx][centrIdx]) ;
                        ++wrong;
                     } else {
                        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) - ( (1-out*yData) * yData * activations[dIdx][centrIdx] ) ) ;
                        ++wrong;
                     }
                  } else {
                     weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) );
                     ++right;
                  }

               }//centrIdx
               
               //bias function
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * (yData - out);

               // calculating second term of objective function               
               if (not settings.L2) 	sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out);
               else 			sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out) * (1 -  yData * out);
                  
               //############ Logging functions ################
               if (out > 0)  { maxOut += out; ++nMax; }
               if (out < 0)  { minOut += out; ++nMin; }
               allOuts[dIdx] = out;
               //###############################################
         

            }//dIdx
            
            //calculate objective function

            // calculating first term of objective function
            float objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }

            float hypPlane = objective;
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack ;
           
            //############ Logging functions ################ 
            maxOuts[svmIdx] = (float)  maxOut / nMax;
            minOuts[svmIdx] = (float)  minOut / nMin;
            avOuts[svmIdx]  = (float) (maxOut + minOut) / nData;
            float stdDevMaxOutPos = 0;
            float stdDevMaxOutNeg = 0;
            float stdDevMinOutPos = 0;
            float stdDevMinOutNeg = 0;
            int nMaxPos = 0;
            int nMaxNeg = 0;
            int nMinPos = 0;
            int nMinNeg = 0;
            for (size_t dIdx=0; dIdx < nData; ++dIdx){
               if (allOuts[dIdx] >  0)   {
                  if (allOuts[dIdx] >  maxOuts[svmIdx]) {stdDevMaxOutPos += pow((allOuts[dIdx] - maxOuts[svmIdx]), 2); ++nMaxPos;}
                  if (allOuts[dIdx] <= maxOuts[svmIdx]) {stdDevMaxOutNeg += pow((allOuts[dIdx] - maxOuts[svmIdx]), 2); ++nMaxNeg;}
               }   
               if (allOuts[dIdx] <= 0)   {
                  if (allOuts[dIdx] >  minOuts[svmIdx]) {stdDevMinOutPos += pow((allOuts[dIdx] - minOuts[svmIdx]), 2); ++nMinPos;}
                  if (allOuts[dIdx] <= minOuts[svmIdx]) {stdDevMinOutNeg += pow((allOuts[dIdx] - minOuts[svmIdx]), 2); ++nMinNeg;}
               }   
            }
            stdDevMaxOutPos = sqrt(stdDevMaxOutPos / nMaxPos);
            stdDevMaxOutNeg = sqrt(stdDevMaxOutNeg / nMaxNeg);
            stdDevMinOutPos = sqrt(stdDevMinOutPos / nMinPos);
            stdDevMinOutNeg = sqrt(stdDevMinOutNeg / nMinNeg);
            statDatFile << itIdx << "," << objective << "," << float (right / (right+wrong) * 100) << "," << minOuts[svmIdx] << "," << maxOuts[svmIdx] << "," << stdDevMinOutPos << "," << stdDevMinOutNeg << "," << stdDevMaxOutPos << "," << stdDevMaxOutNeg << "," << hypPlane / objective * 100 << endl;
            //###############################################

            // online trainings output
            if(normalOut && not debugOut && itIdx % 100 == 0)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t Score = " << (right / (right+wrong) * 100.0) << "\tBias: " << biases[svmIdx] << endl;   
            if(debugOut)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t Score = " << (right / (right+wrong) * 100.0) << "\tBias: " << biases[svmIdx] << endl;   

         }//itIdx
         
        statDatFile.close();  // logfile
         
      }//svmIdx
   }
   

   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector<float>& activations){

      unsigned int maxLabel = 0;
      float maxVal = output(activations, 0);

      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         float out = output(activations, svmIdx);
         if (debugOut) cout << "Output CSVM_" << svmIdx << ": " << out << "\tBias: " << biases[svmIdx] <<endl;
         if(out > maxVal){
            maxVal = out;
            maxLabel = svmIdx;
            if (debugOut) cout << "Maximum output from CSVM_" << maxLabel << "\n" << endl;
         }
      }

      return maxLabel;

   }
