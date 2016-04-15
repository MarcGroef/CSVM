#include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;

/* This settingsfile class parses the settingsfile, and wraps the info up to pass them 
 * to the other classes in the program that require them.
 * 
 * The methods of this class will be called by "CSVMClassifier in csvm_classifier.cc"
 * This is where the actual settings will be passed to the other classes.
 * 
 * The settings are put in a 'struct', so there each time one block of memory passed.
 * These structs are defined in the header files of the class where the're needed.
 * 
 * e.g. csvm_conv_svm.h contains the struct for convSVMSettings;
 * 
 * 
 */

CSVMSettings::~CSVMSettings() {
   //free(analyserSettings.rbmSettings.layerSizes);


}




void CSVMSettings::parseConvSVMSettings(ifstream& stream) {
   string setting;
   string method;
   string value;
   
   stream >> setting;
   if (setting != "loadLastUsed") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> value;
      convSVMSettings.loadLastUsed = (value == "TRUE" || value == "True" || value == "true" || value == "T" || value == "t" || value == "1" || value == "Y" || value == "y");
   }
   
   
   stream >> setting;
   if (setting != "learningRate") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> convSVMSettings.learningRate;

   }

   stream >> setting;
   if (setting != "nIterations") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> convSVMSettings.nIter;
   }

   stream >> setting;
   if (setting != "initWeight") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> convSVMSettings.initWeight;
   }

   stream >> setting;
   if (setting != "CSVM_C") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> convSVMSettings.CSVM_C;
   }

   stream >> setting;
   if (setting != "L2") {
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> value;
      convSVMSettings.L2 = (value == "TRUE" || value == "True" || value == "true" || value == "T" || value == "t" || value == "1" || value == "Y" || value == "y");
   }






}

void CSVMSettings::parseLinNetSettings(ifstream& stream) {


   string setting;
   string method;

   stream >> setting;
   if (setting != "nIterations") {
      cout << "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> netSettings.nIter;

   }

   stream >> setting;
   if (setting != "initWeight") {
      cout << "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> netSettings.initWeight;

   }
   stream >> setting;
   if (setting != "learningRate") {
      cout << "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   else {
      stream >> netSettings.learningRate;
   }


}

void CSVMSettings::parseDatasetSettings(ifstream& stream) {


   string setting;
   string method;
   stream >> setting;
   if (setting != "method") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> method;

   if (method == "CIFAR10") {
      datasetSettings.type = DATASET_CIFAR10;
      stream >> setting;
      if (setting != "nTrainImages") {
         cout << "csvm::csvm_settings:parseDatasetSettings(): In CIFAR10 parsing: Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> datasetSettings.nTrainImages;

      stream >> setting;
      if (setting != "nTestImages") {
         cout << "csvm::csvm_settings:parseDatasetSettings(): In CIFAR10 parsing: Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> datasetSettings.nTestImages;

   }
   else if (method == "MNIST") {
      datasetSettings.type = DATASET_MNIST;
      stream >> setting;
      if (setting != "nTrainImages") {
         cout << "csvm::csvm_settings:parseDatasetSettings(): In MNIST parsing: Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> datasetSettings.nTrainImages;

      stream >> setting;
      if (setting != "nTestImages") {
         cout << "csvm::csvm_settings:parseDatasetSettings(): In MNIST parsing: Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> datasetSettings.nTestImages;
   }
   stream >> setting;
   if (setting != "imageWidth") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> datasetSettings.imWidth;
   
   stream >> setting;
   if (setting != "imageHeight") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> datasetSettings.imHeight;
}


void CSVMSettings::parseCodebookSettings(ifstream& stream) {
   string setting;
   string method;
   stream >> setting;
   if (setting != "generate") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   } else {
      stream >>setting ;
      codebookSettings.generate = (setting == "TRUE" || setting == "True" || setting == "true" || setting == "T" || setting == "t" || setting == "1" || setting == "Y" || setting == "y");
   }
   
   stream >> setting;
   if (setting != "standardize") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   } else {
      stream >>setting ;
      codebookSettings.standardize = (setting == "TRUE" || setting == "True" || setting == "true" || setting == "T" || setting == "t" || setting == "1" || setting == "Y" || setting == "y");
   }
   
   stream >> setting;
   if (setting != "whitening") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   } else {
      stream >>setting ;
      codebookSettings.whitening = (setting == "TRUE" || setting == "True" || setting == "true" || setting == "T" || setting == "t" || setting == "1" || setting == "Y" || setting == "y");
   }
   
   stream >> setting;
   if (setting != "rootNPartitions") {
      cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   } else {
      stream >> codebookSettings.rootNPartitions;
   }
   
   stream >> setting;
   if (setting != "method") {
      cout << "csvm::csvm_settings:parseCodebookSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> method;
   if (method == "LVQ") {
      codebookSettings.method = LVQ_Clustering;

      stream >> setting;
      if (setting == "nClusters") {
         stream >> codebookSettings.lvqSettings.nClusters;
	 codebookSettings.numberVisualWords = codebookSettings.lvqSettings.nClusters; 
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "learningRate") {
         stream >> codebookSettings.lvqSettings.alpha;
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }




   }

   if (method == "KMEANS") {
      codebookSettings.method = KMeans_Clustering;

      stream >> setting;
      if (setting == "nClusters") {
         stream >> codebookSettings.numberVisualWords;
         dcbSettings.nCentroids = codebookSettings.numberVisualWords;

      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "nIterations") {
         stream >> codebookSettings.kmeansSettings.nIter;
         dcbSettings.nIter = codebookSettings.kmeansSettings.nIter;
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "SimilarityFunction") {
         stream >> method;
         if (method == "RBF") {
            codebookSettings.simFunction = CB_RBF;
            dcbSettings.simFunction = DCB_RBF;
         }
         else if (method == "SOFT_ASSIGNMENT") {
            codebookSettings.simFunction = SOFT_ASSIGNMENT;
            dcbSettings.simFunction = DCB_SOFT_ASSIGNMENT;
         }
         else if(method == "COSINE_SOFT_ASSIGNMENT"){
            codebookSettings.simFunction = COSINE_SOFT_ASSIGNMENT;
            dcbSettings.simFunction = DCB_COSINE_SOFT_ASSIGNMENT;
         }
         else
            cout << "Invalid codebook SimilarityFunction: Try RBF or SOFT_ASSIGNMENT\n";
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "similaritySigma") {
         stream >> codebookSettings.similaritySigma;
         dcbSettings.similaritySigma = codebookSettings.similaritySigma;
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

   }

   if (method == "AKMEANS") {
      codebookSettings.method = AKMeans_Clustering;

      stream >> setting;
      if (setting == "nClusters") {
         stream >> codebookSettings.numberVisualWords;
	 
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "nIterations") {
         stream >> codebookSettings.akmeansSettings.nIter;

      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "SimilarityFunction") {
         stream >> method;
         if (method == "RBF")
            codebookSettings.simFunction = CB_RBF;
         else if (method == "SOFT_ASSIGNMENT")
            codebookSettings.simFunction = SOFT_ASSIGNMENT;
         else
            cout << "Invalid codebook SimilarityFunction: Try RBF or SOFT_ASSIGNMENT\n";
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "similaritySigma") {
         stream >> codebookSettings.similaritySigma;
      }
      else {
         cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }


      


   }

}


void CSVMSettings::parseImageScannerSettings(ifstream& stream) {
	string setting;
	string method;
	stream >> setting;
	if (setting == "patchSize") {
		stream >> scannerSettings.patchHeight;
		scannerSettings.patchWidth = scannerSettings.patchHeight;
	}
	else {
		cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
		exit(-1);
	}

	/*
	stream >> setting;
	if (setting == "patchWidth") {
		stream >> scannerSettings.patchWidth;
	}
	else {
		cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
		exit(-1);
	}
	*/
	stream >> setting;
	if (setting == "scanStride") {
		stream >> scannerSettings.stride;
	}
	else {
		cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
		exit(-1);
	}

	stream >> setting;
	if (setting == "nRandomPatches") {
		stream >> scannerSettings.nRandomPatches;
		dcbSettings.nRandomPatches = scannerSettings.nRandomPatches;

	}
	else {
		cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
		exit(-1);
	}

}



void CSVMSettings::parseFeatureExtractorSettings(ifstream& stream) {
	string setting;
	string method;
	string enumeration;
	string useColour;
	stream >> setting;
	if (setting != "method") {
		cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
		exit(-1);
	}
	getline(stream, method);
	//stream >> method;
	cout << "we read:" << method << endl;
	int numOfLFDs = 0;
	string singleFD;
	int LFDnum = 0;
	while (true) {
		singleFD = parseNthToken(method, LFDnum);
		cout << "token: " << LFDnum << " is: " << singleFD << endl;
		if (singleFD == "") {
			cout << "we broke out of read loop" << endl;
			break;
		}
		if (singleFD == "CLEAN" || singleFD == "clean" || singleFD == "raw" || singleFD == "RAW" || singleFD == " CLEAN" || singleFD == "CLEAN ") {
			featureSettings.featureType.insert(featureSettings.featureType.end(), CLEAN);
		}
		if (singleFD == "HOG" || singleFD == "hog" || singleFD == " HOG" || singleFD == "HOG ") {
			featureSettings.featureType.insert(featureSettings.featureType.end(), HOG);
		}
		if (singleFD == "LBP" || singleFD == "lbp" || singleFD == " LBP" || singleFD == "LBP ") {
			featureSettings.featureType.insert(featureSettings.featureType.end(), LBP);
		}

		++LFDnum;
	}

}

string CSVMSettings::parseNthToken(string values, int i) {
	if (values.at(1) == '[') {
		values.erase(values.begin() + 1);
		values.erase(values.end() - 1);
		int valueNo = 0;

		stringstream ss(values);
		string singleFD;
		while (getline(ss, singleFD, ',')) {
			if (singleFD == " ") {

			}
			else {
				if (valueNo == i) {
					return singleFD;
				}
				++valueNo;
			}
		}
		cout << "-1" << endl;
		return "";
	}
	else {
		return values;
	}
}

void CSVMSettings::parseHogSettings(ifstream& stream) {
	string setting;
	string method;
	string enumeration;
	string useColour;
	stream >> setting;
	if (!method.at(0) == '[') {
		//featureSettings.featureType.insert(featureSettings.featureType.end(), HOG);
		//.insert(returnHOG.end(), redCellOrientationHistogram.begin(), redCellOrientationHistogram.end());
		stream >> setting;
		if (setting == "nBins") {  // #nbins is conventionally 9, but can be different.
			stream >> featureSettings.hogSettings[0].nBins;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG nBins not set! Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "cellSize") {  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
			stream >> featureSettings.hogSettings[0].cellSize;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "cellStride") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
			stream >> featureSettings.hogSettings[0].cellStride;

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "patchSize") {  // 
			stream >> featureSettings.hogSettings[0].patchSize;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG patchSize not specified! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "padding") {//#the size of a patch
			stream >> enumeration;
			if (enumeration == "None" || enumeration == "none" || enumeration == "NONE")
				featureSettings.hogSettings[0].padding = NONE;
			else if (enumeration == "Identity" || enumeration == "identity" || enumeration == "IDENTITY")
				featureSettings.hogSettings[0].padding = IDENTITY;
			else if (enumeration == "Zero" || enumeration == "zero" || enumeration == "ZERO")
				featureSettings.hogSettings[0].padding = ZERO;

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "useColourPixel") {//if we use grey images
			stream >> useColour;
			if (useColour == "true" || useColour == "True")
				featureSettings.hogSettings[0].useColourPixel = true;
			else {
				if (useColour == "false" || useColour == "False")
					featureSettings.hogSettings[0].useColourPixel = false;
			}
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "interpolation") {  // things
			stream >> enumeration;
			if (enumeration == "INTERPOLATE_BINARY" || enumeration == "binary" || enumeration == "BINARY" || enumeration == "Binary")
				featureSettings.hogSettings[0].interpol = INTERPOLATE_BINARY;
			else {
				if (enumeration == "INTERPOLATE_LINEAR" || enumeration == "linear" || enumeration == "LINEAR" || enumeration == "Linear")
					featureSettings.hogSettings[0].interpol = INTERPOLATE_LINEAR;
			}

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at interpolationmethod! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "binmethod") {  // things
			stream >> enumeration;
			if (enumeration == "CROSSCOLOUR" || enumeration == "CrossColour" || enumeration == "crosscolour" || enumeration == "Crosscolour")
				featureSettings.hogSettings[0].binmethod = CROSSCOLOUR;
			else {
				if (enumeration == "BYCOLOUR" || enumeration == "ByColour" || enumeration == "bycolour" || enumeration == "Bycolour")
					featureSettings.hogSettings[0].binmethod = BYCOLOUR;
			}

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "postprocessing") {  // things
			stream >> enumeration;
			if (enumeration == "PURE" || enumeration == "Pure" || enumeration == "pure")
				featureSettings.hogSettings[0].postproccess = PURE;
			if (enumeration == "STANDARDISATION" || enumeration == "Standardise" || enumeration == "standardise" || enumeration == "std" || enumeration == "STD")
				featureSettings.hogSettings[0].postproccess = STANDARDISATION;
			if (enumeration == "NORMALISATION" || enumeration == "norm" || enumeration == "Norm" || enumeration == "NORM")
				featureSettings.hogSettings[0].postproccess = NORMALISATION;
			if (enumeration == "LTWONORM" || enumeration == "L2NORM" || enumeration == "L2" || enumeration == "l2norm")
				featureSettings.hogSettings[0].postproccess = LTWONORM;
			if (enumeration == "CLIPNORM" || enumeration == "clipping" || enumeration == "clipnorm" || enumeration == "cnorm")
				featureSettings.hogSettings[0].postproccess = CLIPNORM;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! at HOG postprocessing settings Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "debugLevel") {  // things
			stream >> enumeration;
			if (enumeration == "-1")
				featureSettings.hogSettings[0].debugLevel = -1;
			if (enumeration == "0")
				featureSettings.hogSettings[0].debugLevel = 0;
			if (enumeration == "1")
				featureSettings.hogSettings[0].debugLevel = 1;
			if (enumeration == "2")
				featureSettings.hogSettings[0].debugLevel = 2;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}
	}
}





void CSVMSettings::parseLBPSettings(ifstream& stream) {
	string setting;
	string method;
	string enumeration;
	string useColour;
	stream >> method;
	featureSettings.featureType.insert(featureSettings.featureType.end(), LBP);


		stream >> setting;
		if (setting == "cellSize") {  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
			stream >> featureSettings.lbpSettings[0].cellSize;

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "cellStride") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
			stream >> featureSettings.lbpSettings[0].cellStride;

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "patchSize") {  // 
			stream >> featureSettings.lbpSettings[0].patchSize;
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG patchSize not specified! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}


		stream >> setting;
		if (setting == "padding") {//#the size of a patch
			stream >> enumeration;
			if (enumeration == "None" || enumeration == "none" || enumeration == "NONE")
				featureSettings.lbpSettings[0].padding = LNONE;
			else if (enumeration == "Identity" || enumeration == "identity" || enumeration == "IDENTITY")
				featureSettings.lbpSettings[0].padding = LIDENTITY;
			else if (enumeration == "Zero" || enumeration == "zero" || enumeration == "ZERO")
				featureSettings.lbpSettings[0].padding = LZERO;

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "useColourPixel") {//if we use grey images
			stream >> useColour;
			if (useColour == "true" || useColour == "True")
				featureSettings.lbpSettings[0].useColourPixel = true;
			else {
				if (useColour == "false" || useColour == "False")
					featureSettings.lbpSettings[0].useColourPixel = false;
			}
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "useUniformity") {//if we use grey images
			stream >> enumeration;
			if (enumeration == "UNIFORM" || enumeration == "True" || enumeration == "true")
				featureSettings.lbpSettings[0].uniform = LUNIFORM;
			else {
				if (enumeration == "false" || enumeration == "false" || enumeration == "PURE")
					featureSettings.lbpSettings[0].uniform = LPURE;
			}
		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

		stream >> setting;
		if (setting == "binmethod") {  // things
			stream >> enumeration;
			if (enumeration == "CROSSCOLOUR" || enumeration == "CrossColour" || enumeration == "crosscolour" || enumeration == "Crosscolour")
				featureSettings.lbpSettings[0].binmethod = LCROSSCOLOUR;
			else {
				if (enumeration == "BYCOLOUR" || enumeration == "ByColour" || enumeration == "bycolour" || enumeration == "Bycolour")
					featureSettings.lbpSettings[0].binmethod = LBYCOLOUR;
			}

		}
		else {
			cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
			exit(-1);
		}

}

void CSVMSettings::parseCleanSettings(ifstream& stream) {
	
	string setting;
	string method;
	string enumeration;
	string useColour;
	stream >> method;
	featureSettings.featureType.insert(featureSettings.featureType.end(), CLEAN);

	string type, value;
	stream >> type;
	if (type != "standardize") {
		cout << "csvm::CSVMSettings.parseCleanDescrSettings: Error! Invalid settingsfile layout. Exitting..\n";
		exit(0);
	}
	stream >> value;
	if (value == "None")
		featureSettings.clSettings[0].stdOptions = CL_NONE;
	else if (value == "PER_CHANNEL")
		featureSettings.clSettings[0].stdOptions = CL_PER_CHANNEL;
	else if (value == "ALL")
		featureSettings.clSettings[0].stdOptions = CL_ALL;
	else
		featureSettings.clSettings[0].stdOptions = CL_NONE;
}

void CSVMSettings::parseSVMSettings(ifstream& stream) {
   string setting;
   string method;


   stream >> setting;
   if (setting == "Kernel") {
      stream >> method;
      if (method == "RBF")
         svmSettings.kernelType = RBF;
      else if (method == "LINEAR")
         svmSettings.kernelType = LINEAR;
      else
         cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";

   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   stream >> setting;
   if (setting == "AlphaDataInit") {
      stream >> svmSettings.alphaDataInit;

   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }



   stream >> setting;
   if (setting == "nIterations") {
      stream >> svmSettings.nIterations;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   stream >> setting;
   if (setting == "learningRate") {
      stream >> svmSettings.learningRate;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   stream >> setting;
   if (setting == "SVM_C_Data") {
      stream >> svmSettings.SVM_C_Data;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }


   stream >> setting;
   if (setting == "Cost") {
      stream >> svmSettings.cost;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   stream >> setting;
   if (setting == "D2") {
      stream >> svmSettings.D2;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   stream >> setting;
   if (setting == "sigmaClassicSimilarity") {
      stream >> svmSettings.sigmaClassicSimilarity;
   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }


}

void CSVMSettings::parseGeneralSettings(ifstream& stream) {
   string type, value;

   stream >> type;
   if (type != "Classifier") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;

   if (value == "SVM")
      classifier = CL_SVM;
   else if (value == "CSVM")
      classifier = CL_CSVM;
   else if (value == "LINNET")
      classifier = CL_LINNET;
   else {
      cout << "csvm::parseGeneralSettings: " << value << " is not a recognized classifier method. Exitting..\n";
      exit(0);
   }

   stream >> type;
   if (type != "Codebook") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }

   stream >> value;
   if (value == "CODEBOOK") {
      codebook = CB_CODEBOOK;
   }
   else if (value == "DEEPCODEBOOK") {
      codebook = CB_DEEPCODEBOOK;
   }
   else {
      cout << "csvm::parseGeneralSettings: " << value << " is not a recognized codebook method. Exitting..\n";
      exit(0);
   }
   
   stream >> type;
   if (type != "nCodebooks"){
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> nCodebooks;

   stream >> type;
   if (type != "nClasses") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> netSettings.nClasses;
   convSVMSettings.nClasses = netSettings.nClasses;
   datasetSettings.nClasses = netSettings.nClasses;
   
   stream >> type;
   if (type != "debugOut") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   debugOut = (value == "TRUE");
   
   stream >> type;
   if (type != "normalOut") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   normalOut = (value == "TRUE");

}

/*void CSVMSettings::parseCleanDescrSettings(ifstream& stream){
   string type, value;
   stream >> type;
   if(type != "standardize"){
      cout << "csvm::CSVMSettings.parseCleanDescrSettings: Error! Invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   if(value == "None")
      featureSettings.clSettings.stdOptions = CL_NONE;
   else if(value == "PER_CHANNEL")
      featureSettings.clSettings.stdOptions = CL_PER_CHANNEL;
   else if(value == "ALL")
      featureSettings.clSettings.stdOptions = CL_ALL;
   else
      featureSettings.clSettings.stdOptions = CL_NONE;
}*/

void CSVMSettings::readSettingsFile(string dir) {
   ifstream file(dir.c_str(), ios::in);
   string line;


   if (!file.is_open()) {
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }

   while (getline(file, line) && line != "Dataset");
   parseDatasetSettings(file);
   /*while(getline(file,line) && line != "ClusterAnalyser");
   parseClusterAnalserData(file);*/
   while (getline(file, line) && line != "General");
   parseGeneralSettings(file);
   while (getline(file, line) && line != "Codebook");
   parseCodebookSettings(file);
   while (getline(file, line) && line != "ImageScanner");
   parseImageScannerSettings(file);
   while (getline(file, line) && line != "FeatureExtractor");
   parseFeatureExtractorSettings(file);
   //while(getline(file, line) && line != "CleanDescriptor");
   //parseCleanDescrSettings(file);
   while (getline(file, line) && line != "SVM");
   parseSVMSettings(file);
   while (getline(file, line) && line != "LinNet");
   parseLinNetSettings(file);
   while (getline(file, line) && line != "ConvSVM");
   parseConvSVMSettings(file);
   // parse values:

   file.close();
}

