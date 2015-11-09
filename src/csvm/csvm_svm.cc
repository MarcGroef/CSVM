#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;

/*
 * SVM Constructor: It initializes the SVM.
 * */
SVM::SVM(int datasetSize, int nClusters, int nCentroids, unsigned int labelId){
   //Reserve data for alpha's and set initial values
   alphaData = vector<double>(datasetSize,0.00000000001/*1.0 / datasetSize*/);
   alphaCentroids = vector < vector<double> >(nClusters, vector<double>(nCentroids,1.0 / (nClusters * nCentroids)));
   //Set the class ID of the SVM
   this->classId = labelId;

   //initialize the bias to zero
   bias = 0;
}

/*
 * Aquire parameters from the settings file
 * */
void SVM::setSettings(SVM_Settings s){
   settings = s;
}

/*
 * Updates the AlphaData for the convolutional SVM
 * clActivations: Centroid activations of a given image.
 * dataIdx: Index in the dataset of the given Image
 * */
double SVM::updateAlphaData(vector<Feature>& clActivations, unsigned int dataIdx){
   
   //Aquire the label of the image
   unsigned int dataLabel = clActivations[0].getLabelId();
   
   //Scan the aquired activations for the amount of classes present in the dataset
   unsigned int nClasses = clActivations.size();
   
   //Scan centroid vector of the 0th class for the amount of Visual Words
   unsigned int nCentroids = clActivations[0].content.size();
   
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
         sum += alphaData[dataIdx] * yData * yCentroid * clActivations[cl].content[centr];
      }
   }
   //make the sum negative
   sum = -1.0 * sum;
   //calculate new value for alphaData[dataIdx]
   target = alphaData[dataIdx] + (settings.learningRate * sum);
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
void SVM::constrainAlphaCentroid(vector< vector< Feature > >& activations){

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
      
   }
}

//Constrain the alphaData such that: Sum(alphaData * yCentroid ) == 0, or some threshold
void SVM::contstrainAlphaData(vector< vector< Feature > >& activations, CSVMDataset* ds){
   
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
   for(size_t iter = 0; sum < threshold; ++iter){
      
      //update alphas
      for(size_t dIdx1 = 0; dIdx1 < size; ++dIdx1){
         //yData = ((unsigned int)(activations[dIdx1][0].getLabelId())) == classId ? 1.0 : -1.0;
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
      }
      
   }
}

/*Update the alphas that are related to the centroids
 * 
 * */
double SVM::updateAlphaCentroid(vector< vector< Feature> >& clActivations, unsigned int centrClass, int centr){
   double sum = 0.0;
   unsigned int nData = clActivations.size();
   double yData, yCentroid;
   double target;
   double diff;
   
   yCentroid = (centrClass == classId ? 1.0 : -1.0);
   //calculate sum
   for(size_t dataIdx = 0; dataIdx < nData; ++dataIdx){
      yData = ((clActivations[dataIdx][0].getLabelId()) == classId ? 1.0 : -1.0);
      sum += alphaData[dataIdx] * yData * yCentroid * clActivations[dataIdx][centrClass].content[centr];
   }
   //set new value
   target = alphaCentroids[centrClass][centr] + settings.learningRate * ((double)1.0 - ( (double)sum));
   //keep it in boundaries
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C_Centroid ? settings.SVM_C_Centroid : target;
   diff = alphaCentroids[centrClass][centr] - target;
   //set new value
   alphaCentroids[centrClass][centr] = target;
   return diff;
}

//update alphaData for dual obj. SVM with alpha_i, alpha_j, given a similarity kernal between two activation vectors:
double SVM::updateAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds){
   
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
      
      //calculate the sum
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         yData1 = ((ds->getImagePtr(dIdx1)->getLabelId()) == classId ? 1.0 : -1.0);
         sum += alphaData[dIdx1] * yData0 * yData1 * simKernel[dIdx0].content[dIdx1];
         
      }
      
      //calculate output:
      double output = 0;
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         output += alphaData[dIdx1] * yData1 * simKernel[dIdx0].content[dIdx1];
         
         
      }
      
      deltaAlpha = 1.0 - sum;
      //calc new value
      target = settings.D2 * alphaData[dIdx0] + deltaAlpha * settings.learningRate;
      //keep it in boundaries
      target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
      target = target < 0.0 ? 0.0 : target;
      deltaDiff = alphaData[dIdx0] - target;
      diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
      //set new value
      alphaData[dIdx0] = target;
      
   }
   
   return diff;
}

//make sure  sum(alphaCentroid * yCentroid) == 0, or below threshold
double SVM::constrainAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds){
   
   
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

void SVM::calculateBiasClassic(vector<Feature> simKernel, CSVMDataset* ds){
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
            output += alphaData[dIdx1] * yData * simKernel[dIdx0].content[dIdx1];
            
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
void SVM::trainClassic(vector<Feature> simKernel, CSVMDataset* ds){
   
   double sumDeltaAlpha = 1000.0;
   double prevSumDeltaAlpha = 100.0;
   double deltaAlphaData;

   double convergenceThreshold = 0.0000010 ;

   //for(size_t round = 0; abs(prevSumDeltaAlpha -sumDeltaAlpha) > convergenceThreshold; ++round){
   for(size_t round = 0; /*sumDeltaAlpha > 0.0001*/round < 1500; ++round){
      prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      deltaAlphaData = updateAlphaDataClassic(simKernel, ds);
      deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
      sumDeltaAlpha += deltaAlphaData;
      
      constrainAlphaDataClassic(simKernel, ds);
      if(round % 1000 == 0 )cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << "\tDeltaSOC = " << (prevSumDeltaAlpha - sumDeltaAlpha) << endl;   
      //compute trainings-score:
      
      /*double correct = 0.0;
      for(size_t dIdx0 = 0; dIdx0 < ds->getSize(); ++dIdx0){
         double sum = 0;  
         for(size_t dIdx1 = 0; dIdx1 < ds->getSize(); ++dIdx1){
            double Ydata = ds->getImagePtr(dIdx1)->getLabelId() == classId ? 1.0 : -1.0;
            sum += alphaData[dIdx1] * Ydata * simKernel[dIdx0].content[dIdx1];
         }
         sum += bias;
         if(sum > 0.0 && ds->getImagePtr(dIdx0)->getLabelId() == classId)
            ++correct;
         
      }
      correct /= ds->getSize();
      if(round % 1000 == 0 )cout << "Trainingscore : " << correct << endl;*/
      
      calculateBiasClassic(simKernel, ds);
   }
   
   //for(size_t aIdx = 0; aIdx < alphaData.size(); ++aIdx)
     //    cout << "Alpha " << aIdx << " = " << alphaData[aIdx] << endl;
}


//train the convolutional SVM, given a set of activation for all data-images
void SVM::train(vector< vector<Feature> >& activations, CSVMDataset* ds){
   //read number of classes
   unsigned int nClasses = activations[0].size();
   //read nr of centroids per class
   unsigned int nCentroids = activations[0][0].content.size();
   //read amount of data
   unsigned int size = activations.size();
   
   //set sum of change to artificial high values for initialisation
   double sumDeltaAlpha = 1000.0f;
   //double prevSumDeltaAlpha = 10000.0f;
   double deltaAlphaData, deltaAlphaCentroid;
   
   //set a convergenceThreshold
   double convergenceThreshold = 0.01;
   
   //while the sum of changes in alphas is above threshold:
   for(size_t round = 0; abs(sumDeltaAlpha) > convergenceThreshold; ++round){
      
      //prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      
      //update all alphaData's and count how much they have changed
      /*for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx);
         deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
         sumDeltaAlpha += deltaAlphaData;
      }*/
      //make sure sum(alphaData * yData) is below threshold
      //contstrainAlphaData(activations, ds);
      
      //update all alphaCentroid's
      for(size_t cl = 0;  cl < nClasses; ++cl){
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //update alpha, and measure the change
            deltaAlphaCentroid = updateAlphaCentroid(activations, cl, centr);
            deltaAlphaCentroid = deltaAlphaCentroid < 0.0 ? deltaAlphaCentroid * -1.0 : deltaAlphaCentroid;
            sumDeltaAlpha += deltaAlphaCentroid;
         }
      }
      //make sure sum(alphaCentroid * yData) is below threshold
      //constrainAlphaCentroid(activations);
      cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << endl;   
   }
 
}

//Classify an image, represented by a set of features, using the Convolutional SVM. The codebook pointer is used to get the number of centroids and classes
double SVM::classify(vector<Feature> f, Codebook* cb){
   
   double result = 0;
   
   unsigned int nClasses = cb->getNClasses();
   unsigned int nCentroids = cb->getNCentroids();
   
   double yCentroid;
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         result += alphaCentroids[cl][centr] * yCentroid * f[cl].content[centr];      
      }
   }
   return result;
}

//Classify an image, represented by a set of features, using the classic (alpha_i, alpha_j) SVM
double SVM::classifyClassic(vector<Feature> f, vector< vector<Feature> > datasetActivations, CSVMDataset* ds){
   
   double result = 0;
   unsigned int nData = datasetActivations.size();
   double yData;
   double kernel = 0.0;
   unsigned int nClasses = datasetActivations[0].size();
   unsigned int nCentroids = datasetActivations[0][0].content.size();
   Feature dataKernel(nData,0.0);
   

   
   
   //update sum with similarity between image activations
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      
      double sum = 0;
      //calculate similarity between activation vectors
      
      
      
      
      
      
      //RBF kernel:
      
      /*for(size_t cl = 0;  cl < nClasses; ++cl){
         for(size_t centr = 0; centr < nCentroids; ++centr){
            sum += (datasetActivations[dIdx0][cl].content[centr] - f[cl].content[centr])*(datasetActivations[dIdx0][cl].content[centr] - f[cl].content[centr]);
         }
      }
      //calculate kernel value:
     kernel = exp((-1.0* sum)/settings.sigmaClassicSimilarity);*/
     //cout << "Test kernel value : " << kernel << endl;
      
     
      //Linear kernel
      for(size_t cl = 0; cl < nClasses; ++cl){
         for(size_t centr = 0; centr < nCentroids; ++centr){
            sum += (datasetActivations[dIdx0][cl].content[centr] * f[cl].content[centr]);
         }
      }
      kernel = sum;
      
      
      
     yData = (ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0;
     //add the result for this alpha
     //cout << yData << endl;
     result += alphaData[dIdx0] * yData * kernel;
      
   }
   //add bias to result
   result += bias;
   cout << "SVM " << classId << " has bias " << bias << endl;
   return result;
}