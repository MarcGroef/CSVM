#include <csvm/csvm_svm.h>
#include <algorithm>

using namespace std;
using namespace csvm;

/*
 * SVM Constructor: It initializes the SVM.
 * */
SVM::SVM(int datasetSize, int nClasses, int nCentroids, unsigned int labelId){
   //Reserve data for alpha's and set initial values
   //cout << "Init alpha data " << settings.alphaDataInit << endl;
   this->nCentroids = nCentroids;
   this->datasetSize = datasetSize;
   this->nClasses = nClasses;
   //alphaData = vector<double>(datasetSize,0 /** settings.SVM_C_Data*/);
   //alphaCentroids = vector < vector<double> >(nClasses, vector<double>(nCentroids,0));
   //Set the class ID of the SVM
   this->classId = labelId;

   //initialize the bias to zero
   bias = 0;
}

/*
 * Aquire parameters from the settings file
 * */
void SVM::setSettings(SVM_Settings s){
   //cout << "I recieved " << s.alphaDataInit << endl;
   settings = s;
   alphaData = vector<double>(datasetSize,settings.alphaDataInit /** settings.SVM_C_Data*/);
   alphaCentroids = vector < vector<double> >(nClasses, vector<double>(nCentroids,settings.alphaCentroidInit * settings.SVM_C_Centroid));
}

/*
 * Updates the AlphaData for the convolutional SVM
 * clActivations: Centroid activations of a given image.
 * dataIdx: Index in the dataset of the given Image
 * */
double SVM::updateAlphaData(vector< vector<double> >& clActivations, unsigned int dataIdx, CSVMDataset* ds){
   
   //Aquire the label of the image
   unsigned int dataLabel = ds->getImagePtr(dataIdx)->getLabelId();//clActivations[0].getLabelId();
   
   //Scan the aquired activations for the amount of classes present in the dataset
   unsigned int nClasses = clActivations.size();
   
   //Scan centroid vector of the 0th class for the amount of Visual Words
   unsigned int nCentroids = clActivations[0].size();
   
   //variable to store the change in alphaData[dataIdx] by this update
   double diff;
   
   //variable to store the new value for alphaData[dataIdx]
   double target;
   
   //y-values for data and centroid: 1.0 if the data/centroid label match the SVM label, -1.0 otherwise
   double yData = (dataLabel == classId ? 1.0 : -1.0);
   double yCentroid;
   
   //calculate the summation:
   double sum = 0.0;
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         sum += alphaCentroids[cl][centr] * yData * yCentroid * clActivations[cl][centr];
         //cout << "alp : " << alphaData[dataIdx] << "yData = " << yData << "yCentr = " << yCentroid << ", act = " << clActivations[cl].content[centr] << endl;
      }
   }
   //cout << "sum = " << sum << endl;
   //make the sum negative
   sum = -1.0 * sum;
   //calculate new value for alphaData[dataIdx]
   target = alphaData[dataIdx] + (settings.learningRate * sum);
   //cout << "target = ********************************************************** " << target << endl;
   //make sure that 0 <= alphaData[dataIdx] <= SVM_C_Data
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
   //calculate update difference
   diff = alphaData[dataIdx] - target;
   //set new value
   alphaData[dataIdx] = target;
   //return difference
   return diff;
}

/* Constrain the alphaCentroids such that: Sum(alphaCentroids[cl][centr] * yCentroid ) == 0, or some threshold
 * 
 * 
 * */
void SVM::constrainAlphaCentroid(vector< vector< vector<double> > >& activations){

   unsigned int nClasses = alphaCentroids.size();
   unsigned int nCentroids = alphaCentroids[0].size();
   double yData;
   double oldVal;
   double deltaAlpha;
   double target;
   double treshold = 0.001;
   double sum = 0.0;
   
   //calculate current sum
   for(size_t classIdx0 = 0; classIdx0 < nClasses; ++classIdx0){
         for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
            yData = classIdx0 == classId ? 1.0 : -1.0;
            sum += alphaCentroids[classIdx0][centrIdx] * yData;
         }
   }
   
   //while the sum is still above threshold:
   for(size_t iter = 0; /*iter < nIterations*/ sum > treshold; ++iter){
      
      //adapt deltaAlpha, and update the sum to the new sum:
      for(size_t classIdx0 = 0; classIdx0 < nClasses; ++classIdx0){
         for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
            yData = classIdx0 == classId ? 1.0 : -1.0;
            oldVal = alphaCentroids[classIdx0][centrIdx];
            
            
            deltaAlpha = -2.0 * settings.cost * sum * yData;
            target = alphaCentroids[classIdx0][centrIdx] + deltaAlpha * settings.learningRate;
            target = target < 0.0 ? 0.0 : target;
            target = target > settings.SVM_C_Centroid ? settings.SVM_C_Centroid : target;
            
            alphaCentroids[classIdx0][centrIdx] = target;
            sum += ( alphaCentroids[classIdx0][centrIdx] - oldVal ) * yData;
         }
      }
      //cout << "constrCentrIter\n";
   }
}

//Constrain the alphaData such that: Sum(alphaData * yCentroid ) == 0, or some threshold
void SVM::contstrainAlphaData(vector< vector< vector<double> > >& activations, CSVMDataset* ds){
   
   unsigned int size = alphaData.size();
   double yData;
   double oldVal;
   double deltaAlpha;
   double target;
   double threshold = 0.001;
   double sum = 0;
   
   //calculate current sum
   for(size_t dIdx0 = 0; dIdx0 < size; ++dIdx0){
         yData = (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0;
         sum += alphaData[dIdx0] * yData;
   }
   
   //while the sum is above threshold:
   for(size_t iter = 0; sum > threshold; ++iter){
      
      //update alphas
      for(size_t dIdx1 = 0; dIdx1 < size; ++dIdx1){
         yData = ((unsigned int)(ds->getImagePtr(dIdx1)->getLabelId())) == classId ? 1.0 : -1.0;
         oldVal = alphaData[dIdx1];
         deltaAlpha = -2.0 * settings.cost * sum * yData;
         //calc new value
         target = alphaData[dIdx1] + deltaAlpha * settings.learningRate;
         //keep them within boundaries
         target = target < 0.0 ? 0.0 : target;
         target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
         
         alphaData[dIdx1] = target;
         //adjust sum
         sum += ( alphaData[dIdx1] - oldVal ) * yData;
         //cout << "sum = " << sum << endl;
      }
      //cout << "constrAlpData\n";
   }
}

/*Update the alphas that are related to the centroids
 * 
 * */
double SVM::updateAlphaCentroid(vector< vector< vector< double> > >& clActivations, unsigned int centrClass, int centr, CSVMDataset* ds){
   double sum = 0.0;
   unsigned int nData = clActivations.size();
   double yData, yCentroid;
   double target;
   double diff;
   double C = 10;
   yCentroid = (centrClass == classId ? 1.0 : -1.0);
   //calculate sum
   for(size_t dataIdx = 0; dataIdx < nData; ++dataIdx){
      //yData = ((clActivations[dataIdx][0].getLabelId()) == classId ? 1.0 : -1.0);
      yData = (ds->getImagePtr(dataIdx)->getLabelId() == classId ? 1.0 : -1.0);
      //if(yData > -1.0) cout << "ydata " << yData << " yCentr " << yCentroid << endl;
      //sum += alphaData[dataIdx] * yData * yCentroid * clActivations[dataIdx][centrClass][centr];
      //cout << "alphaData " << alphaData[dataIdx] << endl;
      sum += (alphaData[dataIdx] + C * yData) * clActivations[dataIdx][centrClass][centr];
      //cout << "Activ: " <<clActivations[dataIdx][centrClass].content[centr] << endl;
   }
   //cout << "Ycentr = " << yCentroid << " sum = " << sum << endl;
  //cout << "sum " << sum << endl;
   //set new value
   //target = sum;
   double deltaAlphaCentroid = yCentroid * sum;
   target = alphaCentroids[centrClass][centr] + settings.learningRate * deltaAlphaCentroid;
   //target = (1- settings.learningRate ) * alphaCentroids[centrClass][centr] + settings.learningRate * sum;//((double)1.0- ( (double)sum));
   //keep it in boundaries
   //target = target < 0.0 ? 0.0 : target;
   //cout << "target = " << target << endl;
   target = target > settings.SVM_C_Centroid ? settings.SVM_C_Centroid : target;
   target = target < -1 * settings.SVM_C_Centroid ? -1 * settings.SVM_C_Centroid : target;
   diff = alphaCentroids[centrClass][centr] - target;
   //set new value
   alphaCentroids[centrClass][centr] = target;
   return diff;
}

//update alphaData for dual obj. SVM with alpha_i, alpha_j, given a similarity kernel between two activation vectors:
double SVM::updateAlphaDataClassic(vector< vector< double >  >& simKernel, CSVMDataset* ds){
   
   double diff = 0.0;
   double deltaDiff = 0.0;
   double target;
   double sum = 0.0;
   double yData0;
   double yData1;
   unsigned int nData = simKernel.size();
   double deltaAlpha;

   //for all alpha's:
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      deltaDiff = 0.0;
      yData0 = ((ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0);
      sum = 0.0;
      //calculate the sum
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         yData1 = ((ds->getImagePtr(dIdx1)->getLabelId()) == classId ? 1.0 : -1.0);
         sum += alphaData[dIdx1] * yData0 * yData1 * simKernel[dIdx0][dIdx1];
         //cout << "alphaD = " << alphaData[dIdx1] << " Yi = " << yData0 << " Yj = " << yData1 <<" simkern = " << simKernel[dIdx0][dIdx1] << endl;
         
      }
      //cout << "sum = " << sum << endl;
      //calculate output:
      /*double output = 0;
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         output += alphaData[dIdx1] * yData1 * simKernel[dIdx0].content[dIdx1];
         
         
      }*/
      
      deltaAlpha = 1.0 - sum;
      //calc new value
      target = settings.D2 * alphaData[dIdx0] + deltaAlpha * settings.learningRate;
      //keep it in boundaries
      
      target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
      /*if(settings.kernelType != LINEAR)*/ target = target < 0.0 ? 0.0 : target;
      //else target = target < settings.SVM_C_Data ? settings.SVM_C_Data : target;
      deltaDiff = alphaData[dIdx0] - target;
      diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
      //set new value
      alphaData[dIdx0] = target;
      
   }
   
   return diff;
}

//make sure  sum(alphaCentroid * yCentroid) == 0, or below threshold
double SVM::constrainAlphaDataClassic(vector< vector<double> >& simKernel, CSVMDataset* ds){
   
   
   double oldVal;
   unsigned int nData = simKernel.size();
   double deltaAlpha;
   double yData; 
   double target;
   
   double diff = 0.0;
   double deltaDiff = 0.0;
   double sum = 0;
   double threshold = 0.01;
   
   //calculate current sum
   for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         yData = (classId == (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         sum += alphaData[dIdx0] * yData;
   }
   
   //while sum is above threshold, update alphas
   for(size_t constItr = 0; sum > threshold; ++constItr){
      for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         
         yData = (classId == (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         oldVal = alphaData[dIdx0];
         deltaAlpha = -2.0 * settings.cost * sum * yData;
         
         //make sure that 0 <= alphaCentroid <= SVM_C_Centroid
         target = alphaData[dIdx0] + deltaAlpha * settings.learningRate;
         target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
         target = target < 0.0 ? 0.0 : target;
         deltaDiff = alphaData[dIdx0] - target;
         diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
         
         alphaData[dIdx0] = target;
         sum += (alphaData[dIdx0] - oldVal) * yData;
      }
   }
   return diff;
}

void SVM::calculateBiasClassic(vector< vector< double> >& simKernel, CSVMDataset* ds){
   bias = 0;
   unsigned int total = 0;
   unsigned int nData = simKernel.size();
   double output;
   double yData;
   
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      //if((alpha_coeff[C][i] > 0.0) && ((alpha_coeff[C][i]) < (SVM_C - 0.000001)  Marco 
      if((alphaData[dIdx0] > 0) && (alphaData[dIdx0] < settings.SVM_C_Data - 0.000001)){
         output = 0;
         for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
            yData = ((unsigned int)(ds->getImagePtr(dIdx1)->getLabelId()) == classId ? 1.00 : -1.00);
            output += alphaData[dIdx1] * yData * simKernel[dIdx0][dIdx1];
            
         }
         yData = ((unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.00 : -1.00);
         bias += yData - output;
         ++total;
      }
   }
   
   if(total == 0)
      bias = 0;
   else 
      bias /= total;
}
void SVM::trainClassic(vector< vector< double> >& simKernel, CSVMDataset* ds){
   
   double sumDeltaAlpha = 1000.0;
   double prevSumDeltaAlpha = 100.0;
   double deltaAlphaData;

   double convergenceThreshold = 0.001 ;

   //for(size_t round = 0; abs(prevSumDeltaAlpha -sumDeltaAlpha) > convergenceThreshold; ++round){
   double prevObjective = 0.0;
   double objective = 0.0;
   double sum0 = 0.0;
   double sum1 = 0.0;
   for(size_t round = 0; /*sumDeltaAlpha > 0.00001*/ (prevObjective - objective < -0.0001 || round < 100)&& round < settings.nIterations; ++round){
      prevObjective = objective;
      prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      deltaAlphaData = updateAlphaDataClassic(simKernel, ds);
      deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
      sumDeltaAlpha += deltaAlphaData;
      
      constrainAlphaDataClassic(simKernel, ds);
      
      
      double yData0, yData1;
      
      //calculate objective
      sum0 = 0.0;
      sum1 = 0.0;
      for(size_t dIdx0 = 0; dIdx0 < simKernel.size(); ++dIdx0){
         yData0 = (ds->getImagePtr(dIdx0)->getLabelId() == classId ? 1.0 : -1.0);
         
         sum0 += alphaData[dIdx0];
         for(size_t dIdx1 = 0; dIdx1 < simKernel.size(); ++dIdx1){
            yData1 = (ds->getImagePtr(dIdx1)->getLabelId() == classId ? 1.0 : -1.0);
            
            sum1 += alphaData[dIdx0] * alphaData[dIdx1] * yData0 *  yData1 * simKernel[dIdx0][dIdx1];
            
         }
      }
      objective = sum0 - 0.5 * sum1;
         
     if(round % 100 == 0 )cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << "\tDeltaSOC = " << (prevSumDeltaAlpha - sumDeltaAlpha) << "  Objective : " << objective << endl;   
      //compute trainings-score:
      
   }
   calculateBiasClassic(simKernel, ds);
   //for(size_t aIdx = 0; aIdx < alphaData.size(); ++aIdx)
     //    cout << "Alpha " << aIdx << " = " << alphaData[aIdx] << endl;
}

bool wayToSort(double i, double j) { return i > j; }
//train the convolutional SVM, given a set of activation for all data-images
void SVM::train(vector< vector< vector< double > > >& activations, CSVMDataset* ds){
   //read number of classes
   unsigned int nClasses = activations[0].size();
   //read nr of centroids per class
   unsigned int nCentroids = activations[0][0].size();
   //read amount of data
   unsigned int size = activations.size();
   
   //set sum of change to artificial high values for initialisation
   double sumDeltaAlpha = 1000.0f;
   //double prevSumDeltaAlpha = 10000.0f;
   double deltaAlphaData, deltaAlphaCentroid;
   
   //set a convergenceThreshold
   double convergenceThreshold = 0.01;
   double learnInit = settings.learningRate;
   double prevObj = 0;
   double obj = 0;
   //while the sum of changes in alphas is above threshold:
   //for(size_t round = 0; /*sumDeltaAlpha > 0.00001 && round < settings.nIterations*/ prevObj - obj > 0.0 || round < 9; ++round){
      prevObj = obj;
      //prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      
      //inspect activations:
      /*for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         cout << "*****\n";
         for(size_t cl = 0;  cl < nClasses; ++cl){
            for(size_t centr = 0; centr < nCentroids; ++centr){
               cout << activations[dataIdx][cl].content[centr] << endl;
            }
         }
      }*/
      
      //inspection for marco
      /*for(size_t dIdx = 0; dIdx < size; ++dIdx){
         
         if(ds->getImagePtr(dIdx)->getLabelId() == classId){
            cout << "Inspection image " << dIdx << " for svm " << classId << endl;
            vector<double> kernelArray(nClasses * nCentroids);
            for(size_t cl = 0;  cl < nClasses; ++cl)
               for(size_t centr = 0; centr < nCentroids; ++centr){
                  kernelArray[cl * nCentroids + centr] = activations[dIdx][cl][centr];
               }
            sort(kernelArray.begin(), kernelArray.end(), wayToSort);
            for(size_t kIdx = 0; kIdx < 10; ++kIdx){
               int yP = 0;
               
               for(size_t cl = 0;  cl < nClasses; ++cl)
                  for(size_t centr = 0; centr < nCentroids; ++centr){
                     if(kernelArray[kIdx] == activations[dIdx][cl][centr]){
                        yP = ( cl == classId ? 1 : -1);
                        
                     }
                     if(yP != 0)
                        break;
                  }
               //kernelArray[kIdx] *= yP;
               cout << "top " << kIdx << " : kernelValue * y_p = " << kernelArray[kIdx] << " * " << yP << " = " << kernelArray[kIdx] * yP  << endl;
            }
         }
      }*/
      
      //update all alphaData's and count how much they have changed
      /*for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
       * 
       * 
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx, ds);
         //cout << "alphaData[" << dataIdx << "] = " << alphaData[dataIdx] << endl;
         deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
         sumDeltaAlpha += deltaAlphaData;
      }
      //make sure sum(alphaData * yData) is below threshold
      //contstrainAlphaData(activations, ds);
      */
      
      //update all alphaCentroid's
      for(size_t cl = 0;  cl < nClasses; ++cl){
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //update alpha, and measure the change
            deltaAlphaCentroid = updateAlphaCentroid(activations, cl, centr, ds);
            deltaAlphaCentroid = deltaAlphaCentroid < 0.0 ? deltaAlphaCentroid * -1.0 : deltaAlphaCentroid;
            sumDeltaAlpha += deltaAlphaCentroid;
         }
      }
      
      vector<double> out(alphaData.size(),0);
      double epsilon = 0.1;
      //set alpha_data to 0 is nessesary
      for(size_t dIdx = 0; dIdx < alphaData.size(); ++dIdx){
         out[dIdx] = 0;
         double yData = (ds->getImagePtr(dIdx)->getLabelId() == classId ? 1.0 : -1.0);
         
         for(size_t cl = 0;  cl < nClasses; ++cl){
            for(size_t centr = 0; centr < nCentroids; ++centr){
               double yCentr = (cl == classId ? 1.0 : -1.0);
               out[dIdx] += alphaCentroids[cl][centr] * yCentr * activations[dIdx][cl][centr];
            }
         }
         
         alphaData[dIdx] = alphaData[dIdx] + settings.learningRate * ( 1 - epsilon - out[dIdx] * yData);
         if(alphaData[dIdx] < 0) alphaData[dIdx] = 0;
         if(alphaData[dIdx] > settings.SVM_C_Data) alphaData[dIdx] = settings.SVM_C_Data;
      }
      
      //make sure sum(alphaCentroid * yData) is below threshold
     // constrainAlphaCentroid(activations);
      
      //Measure 
      double yData, yCentroid;
      obj = 0;
      double sum0 = 0.0;
      for(size_t cl = 0;  cl < nClasses; ++cl){
            for(size_t centr = 0; centr < nCentroids; ++centr){
               sum0 += alphaCentroids[cl][centr]*alphaCentroids[cl][centr];
            }
      }
      double sum1 = 0.0;
      for(size_t dIdx = 0; dIdx < alphaData.size(); ++dIdx){
         
         yData = (ds->getImagePtr(dIdx)->getLabelId() == classId ? 1.0 : -1.0);
      
         
               
         sum1 += alphaData[dIdx] * (1 - epsilon - out[dIdx] * yData);
         
      }
         
      obj = 0.5*sum0 +  sum1;
      //cout << "obective == " << obj << endl;
      //settings.learningRate = (1.0 / round < 0.0001 ? 0.0001 : 1.0/round);
      //settings.learningRate = learnInit*(settings.nIterations - round ) / settings.nIterations;
      
      
      cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << " Objective : " << obj << endl;   
  // }
 
}

//Classify an image, represented by a set of features, using the Convolutional SVM. The codebook pointer is used to get the number of centroids and classes
double SVM::classify(vector< vector< double > >  f, Codebook* cb){
   
   double result = 0;
   
   unsigned int nClasses = cb->getNClasses();
   unsigned int nCentroids = cb->getNCentroids();
   
   double yCentroid;


   
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         //cout << "yCentr = " << yCentroid << " alpha = " << alphaCentroids[cl][centr] << endl;
         result += alphaCentroids[cl][centr] * yCentroid * f[cl][centr];  
      }
   }
   //cout << "SVM " << classId << " says: " << result << endl; 
   return result;
   
}

//Classify an image, represented by a set of features, using the classic (alpha_i, alpha_j) SVM
double SVM::classifyClassic(vector< vector< double > > f, vector< vector< vector<double> > >& datasetActivations, CSVMDataset* ds){
   
   double result = 0;
   unsigned int nData = datasetActivations.size();
   double yData;
   double kernel = 0.0;
   unsigned int nClasses = datasetActivations[0].size();
   unsigned int nCentroids = datasetActivations[0][0].size();
   Feature dataKernel(nData,0.0);
   

   
   
   //update sum with similarity between image activations
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      
      double sum = 0;
      //calculate similarity between activation vectors
      
      
      //RBF kernel:
      if(settings.kernelType == RBF){
         for(size_t cl = 0;  cl < nClasses; ++cl){
            for(size_t centr = 0; centr < nCentroids; ++centr){
               //sum of distance squared
               sum += (datasetActivations[dIdx0][cl][centr] - f[cl][centr])*(datasetActivations[dIdx0][cl][centr] - f[cl][centr]);
            }
         }
         //calculate kernel value:
         kernel = exp((-1.0* sum)/settings.sigmaClassicSimilarity);
         //cout << "Test kernel value : " << kernel << endl;
      }else if (settings.kernelType == LINEAR){
     
         //Linear kernel
         for(size_t cl = 0; cl < nClasses; ++cl){
            for(size_t centr = 0; centr < nCentroids; ++centr){
               sum += (datasetActivations[dIdx0][cl][centr] * f[cl][centr]);
            }
         }
         kernel = sum;
      }else{
         cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
      }
      
      
      
     yData = (ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0;
     //add the result for this alpha
     //cout << yData << endl;
     result += alphaData[dIdx0] * yData * kernel;
      
   }
   //add bias to result
   result += bias;
   //cout << "SVM " << classId << " has bias " << bias << endl;
   //cout << "SVM " << classId << " says " << result << endl;
   return result;
}