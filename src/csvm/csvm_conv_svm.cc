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

//unions for binary export;
union charInt{
   char chars[4];
   unsigned int intVal;
};

union charDouble{
   char chars[8];
   double doubleVal;
};

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      
      
   }
	
   void ConvSVM::initialize(){
      weights = vector< vector<double> >(settings.nClasses, vector<double>(settings.nCentroids, settings.initWeight));
      biases  = vector<double>(settings.nClasses, 0);
   }

   // classification given an activation vector
   double ConvSVM::output(vector<double>& activations, unsigned int svmIdx){
      
      unsigned int nCentroids = activations.size();
      //cout << "nCentroid@out = " << nCentroids << " at svm " << svmIdx << endl;
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         out += weights[svmIdx][centrIdx] * activations[centrIdx];
      }
      
      out += biases[svmIdx];
      return out;
   }
   


   // training
   void ConvSVM::train(vector< vector<double> >& activations, unsigned int nIterations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      cout << fixed << setprecision(5);


      //############ Logging functions ################
      maxOuts = vector<double>(settings.nClasses, 0);
      minOuts = vector<double>(settings.nClasses, 0);
      avOuts  = vector<double>(settings.nClasses, 0);
      ofstream statDatFile;
      //###############################################
      //for all training iterations
      for(size_t itIdx = 0; itIdx < nIterations; ++itIdx){
      
         //############ Logging functions ################
         //stringstream ss;
        // ss << "statData_SVM-" << svmIdx << ".csv";
        // string fName = ss.str();
       //  statDatFile.open ( fName.c_str() );
       //  statDatFile << "Iteration,Objective,Score,MinOut,MaxOut,stdDevMinOutPos,stdDevMinOutNeg,StdDevMaxOutPos,StdDevMaxOutNeg,HyperplanePercentage" << endl;
        // if (debugOut) cout << "\n\nSVM " << svmIdx << ":\t\t\t(data written to " << fName << ")\n" << endl;
         //###############################################
         //for all csvms in ensemble
	 for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         
            
            double sumSlack = 0;
            float right = 0;
            float wrong = 0;

            //############ Logging functions ################
            allOuts  = vector<double>(nData, 0);
            minOut = 0;
            maxOut = 0;
            nMax   = 0;
            nMin   = 0;
            //###############################################
         

            // for all datapoints
            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               unsigned int label = ds->getTrainImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
	       
               // for all centers 
               for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
                  // partial derivatives to the weights
                  if(yData * out < 1){
                     if (! settings.L2){
                        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) -  yData * activations[dIdx][centrIdx]) ;
                        //++wrong;
                     } else {
                        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) - ( (1-out*yData) * yData * activations[dIdx][centrIdx] ) ) ;
                        //++wrong;
                     }
                  } else {
                     weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) );
                     //++right;
                  }

               }//centrIdx
               
               //bias function
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * (yData - out);

               // calculating second term of objective function               
               if (! settings.L2) 	sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out);
               else 			sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out) * (1 -  yData * out);
                  
               //############ Logging functions ################
               //if (out > 0)  { maxOut += out; ++nMax; }
               //if (out < 0)  { minOut += out; ++nMin; }
               //allOuts[dIdx] = out;
               //###############################################
         

            }//dIdx
            
            //calculate objective function

            // calculating first term of objective function
            /*double objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }

            double hypPlane = objective;
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack ;
           
            //############ Logging functions ################ 
            maxOuts[svmIdx] = (double)  maxOut / nMax;
            minOuts[svmIdx] = (double)  minOut / nMin;
            avOuts[svmIdx]  = (double) (maxOut + minOut) / nData;
            double stdDevMaxOutPos = 0;
            double stdDevMaxOutNeg = 0;
            double stdDevMinOutPos = 0;
            double stdDevMinOutNeg = 0;
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
*/
            // online trainings output
            //if(normalOut && not debugOut && itIdx % 100 == 0)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t Score = " << (right / (right+wrong) * 100.0) << "\tBias: " << biases[svmIdx] << endl;   
            //if(debugOut)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t Score = " << (right / (right+wrong) * 100.0) << "\tBias: " << biases[svmIdx] << endl;   
	    //if(normalOut && not debugOut && itIdx % 100 == 0)cout << "CSVM " << svmIdx << " : " << itIdx << " / " << settings.nIter << endl;
          }//itIdx
         
        //statDatFile.close();  // logfile
         
      }//svmIdx
   }
   

   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector<double>& activations){

      unsigned int maxLabel = 0;
      double maxVal = output(activations, 0);

      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         if (debugOut) cout << "Output CSVM_" << svmIdx << ": " << out << "\tBias: " << biases[svmIdx] <<endl;
         if(out > maxVal){
            maxVal = out;
            maxLabel = svmIdx;
            if (debugOut) cout << "Maximum output from CSVM_" << maxLabel << "\n" << endl;
         }
      }

      return maxLabel;

   }
   
   double ConvSVM::validate(vector< vector<double> >& validationActivations, CSVMDataset* dataset){
      size_t nTestImages = dataset->getTestSize();
      unsigned int nCorrect = 0;
      unsigned int result;
      
      for(size_t dIdx = 0; dIdx != nTestImages; ++dIdx){
	result = classify(validationActivations[dIdx]);
	if(result == dataset->getTestImagePtr(dIdx)->getLabelId())
	  ++nCorrect;
      }
      return (double)nCorrect / nTestImages;
   }
   
   void ConvSVM::exportToFile(string fname){
      //first: int with number of outputNodes;
      //second: int with number of weigths per node;
      //third: bias per node
      //fourth: weigths per node
      
      charInt fancyInt;
      charDouble fancyDouble;
   
      ofstream file(fname.c_str(), ios::binary);
      
      //write nr of output nodes
      fancyInt.intVal = settings.nClasses;
      file.write(fancyInt.chars, 4);
      
      //write nWeightsPerNode
      fancyInt.intVal = settings.nCentroids;
      file.write(fancyInt.chars, 4);
      
      //write biases
      for(size_t nIdx = 0; nIdx != settings.nClasses; ++nIdx){
         fancyDouble.doubleVal = biases[nIdx];
         file.write(fancyDouble.chars, 8);
         
      }
      
      //write weights
      for(size_t nIdx = 0; nIdx != settings.nClasses; ++nIdx){
         for(size_t cIdx = 0; cIdx != settings.nCentroids; ++cIdx){
            fancyDouble.doubleVal = weights[nIdx][cIdx];
            file.write(fancyDouble.chars, 8);
         }
      }
      
      file.close();
      
      
   }
   
   void ConvSVM::importFromFile(string fname){
      //first: int with number of outputNodes;
      //second: int with number of weigths per node;
      //third: bias per node
      //fourth: weigths per node
      
      charInt fancyInt;
      charDouble fancyDouble;
      
      ifstream file(fname.c_str(), ios::binary);
      
      //read nNodes
      file.read(fancyInt.chars, 4);
      settings.nClasses = fancyInt.intVal;
      
      //read nr of weights per node
      file.read(fancyInt.chars, 4);
      settings.nCentroids = fancyInt.intVal;
      
      //prepare/reset bias vector
      biases = vector<double>();
      
      //read biases
      for(size_t nIdx = 0; nIdx != settings.nClasses; ++nIdx){
         file.read(fancyDouble.chars, 8);
         biases.push_back(fancyDouble.doubleVal);
      }
      
      //prepare/reset weights
      weights = vector< vector<double> >();
      
      for(size_t nIdx = 0; nIdx != settings.nClasses; ++nIdx){
         vector<double> nodeWeights;
         for(size_t cIdx = 0; cIdx != settings.nCentroids; ++cIdx){
            file.read(fancyDouble.chars, 8);
            nodeWeights.push_back(fancyDouble.doubleVal);
         }
         weights.push_back(nodeWeights);
      }
      file.close();
   }