union charDouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

void MLPController::importFeatureSet(string filename){
   charInt fancyInt;
   charDouble fancyDouble;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::binary);
   
   //read number of classes
   file.read(fancyInt.chars,4);
   nClasses = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << nClasses << " classes\n";
   
   //read nr of visual words
   file.read(fancyInt.chars, 4);
   settings.numberVisualWords = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << settings.numberVisualWords << " words per class\n";
   //read typesize
   char c;
   file.read(&c,1);
   typesize = c;
   //let the compiler shutup about the fact that there is no dynamic type support yet
   ++typesize;
   --typesize;
   
   //read feature dimensionality
   file.read(fancyInt.chars, 4);
   
   featDims = fancyInt.intVal;
   //allocate space

   //read centroids

   for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
      //Feature f(featDims,0);
      Centroid c;
      c.content.resize(featDims);
      for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
         file.read(fancyDouble.chars, 8);
         c.content[featIdx] = fancyDouble.doubleVal;
      }
      bow.push_back(c);
   }
   
   
   file.close();
}

void MLPController::exportFeatureSet(string filename, vector<Feature>& featureSet){
   /* Featureset file conventions:
    * 
    * first,  Dataset			: 0, CIFAR10 1,MNIST   (1 byte)
    * second, Amount of features: 0-10.000.000         (4 bytes)
    * third,  PatchWidth		: 0-36                 (1 byte)
    * fourth, PatchHeigth		: 0-36                 (1 byte)  
    * fifth,  FeatSize			: 0-10.000             (1 byte)
    * sixth,  FeatExtractor		: 0,HOG 1,CLEAN        (1 byte)  
    * 
    * Now each lines consists all the double values in the feature
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charDouble fancyDouble;
   
   //cout << "\t\twordSize:\t" << bow[0].content.size() << "\n\tfilename:\t" << filename.c_str() << endl;
   //unsigned int wordSize = bow[0].content.size();
   ofstream file(filename.c_str(),  ios::binary);
   
   //write dataset used
   fancyInt.intVal = settings.datasetSettings.method = "CIFAR10"?0:1; //is this correct?
   file.write(fancyInt.chars, 1);
   
   //write amount of features
   fancyInt.intVal = settings.scannerSettings.nRandomPatches;
   file.write(fancyInt.chars, 4);
   
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchWidth;
   file.write(fancyInt.chars, 1);
   
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchHeigth;
   file.write(fancyInt.chars, 1);
   
   //write FeatSize
   fancyInt.intVal = settings.mlpSettings.nInputUnits;
   file.write(fancyInt.chars, 1);   
   
   //write FeatExtractor method
   fancyInt.intVal = settings.FeatureExtractorSettings.method = "HOG"?0:1;
   file.write(fancyInt.chars, 1); 
   
   for(size_t word = 0; word < settings.numberVisualWords; ++word){
      for (size_t val = 0; val < wordSize; ++val){
         fancyDouble.doubleVal = bow[word].content[val];
         file.write(fancyDouble.chars, 8);
      }
   } 
   
	for(int i=0;i<settings.scannerSettings.nRandomPatches;i++){
		for(int j=0;j<settings.mlpSettings.nInputUnits;j++){
			fancyDouble.doubleVal = featureSet[i].content[j];
			file.write(fancyDouble.chars, 8);
		}
	}
   
   file.close();
}

void MLPController::importFeatureSet(string filename){
   charInt fancyInt;
   charDouble fancyDouble;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::binary);
   
   std::cout << file.read(fancyInt.chars,1) << std::endl;
   std::cout << file.read(fancyInt.chars,4) << std::endl;
   std::cout << file.read(fancyInt.chars,1) << std::endl;
   std::cout << file.read(fancyInt.chars,1) << std::endl;
   
   //read number of classes
   file.read(fancyInt.chars,4);
   nClasses = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << nClasses << " classes\n";
   
   //read nr of visual words
   file.read(fancyInt.chars, 4);
   settings.numberVisualWords = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << settings.numberVisualWords << " words per class\n";
   //read typesize
   char c;
   file.read(&c,1);
   typesize = c;
   //let the compiler shutup about the fact that there is no dynamic type support yet
   ++typesize;
   --typesize;
   
   //read feature dimensionality
   file.read(fancyInt.chars, 4);
   
   featDims = fancyInt.intVal;
   //allocate space

   //read centroids

   for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
      //Feature f(featDims,0);
      Centroid c;
      c.content.resize(featDims);
      for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
         file.read(fancyDouble.chars, 8);
         c.content[featIdx] = fancyDouble.doubleVal;
      }
      bow.push_back(c);
   }
   file.close();
}
