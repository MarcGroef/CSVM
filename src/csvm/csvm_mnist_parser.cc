#include <csvm/csvm_mnist_parser.h>
#include <stdint.h>

using namespace std;
using namespace csvm;

   void swapByte(unsigned char* byte){
      unsigned char newByte = 0;
      unsigned char bit;
      for(size_t bIdx = 0; bIdx < 8; ++ bIdx){
         bit = ((1 << bIdx) & *byte);
         newByte |= bit << (7 - bIdx);
      }
      *byte = newByte;
   }
   
   MNISTParser::MNISTParser(){
      trainImages = NULL;
      trainLabels = NULL;
      testImages = NULL;
      testLabels = NULL;
      
      trainImagesFile = "train-images.idx3-ubyte";
      trainLabelFile = "train-labels.dix1-ubyte";
      testImagesFile = "t10k-images.idx3-ubyte";
      testLabelFile = "t10k-labels.idx3-ubyte";
   }
   
   void MNISTParser::readTrainImages(string dir){
      basic_ifstream<unsigned char> file((dir + trainImagesFile).c_str(),ios::in|ios::binary|ios::ate);
      trainImages = new MNISTTrainSet;
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTrainImages(" << (dir + trainImagesFile) << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
      cout << "filesize = " << size << endl;
      
      file.seekg(0,ios::beg);
      file.read(trainImages->data,size);
      cout << "magic = " << (unsigned int*)(trainImages->data)[0] << endl;;
      if(trainImages->formatted.magicNumber != 2051)
         cout << "csvm::MNISTParser::readTrainImages: Magic number mismatch! It says " << trainImages->formatted.magicNumber << " and reads " << trainImages->formatted.numberOfImages << " images\n";
      
      file.close();
   }
   
   void MNISTParser::readTrainLabels(string dir){
      basic_ifstream<unsigned char> file((dir + trainLabelFile).c_str(),ios::in|ios::binary|ios::ate);
      trainLabels = new MNISTTrainLabels;     
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTrainLabels(" << (dir + trainLabelFile) << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(trainLabels->data,size);
      
      if(trainLabels->formatted.magicNumber != 2049)
         cout << "csvm::MNISTParser::readTrainLabels: Magic number mismatch! It says " << trainLabels->formatted.magicNumber<< "\n";
      
      file.close();
   }
   
   void MNISTParser::readTestImages(string dir){
      basic_ifstream<unsigned char> file((dir + testImagesFile).c_str(),ios::in|ios::binary|ios::ate);
      testImages = new MNISTTestSet;    
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTestImages(" << (dir + testImagesFile) << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(testImages->data,size);
      
      if(trainLabels->formatted.magicNumber != 2051)
         cout << "csvm::MNISTParser::readTestImages: Magic number mismatch!\n";
      
      
      file.close();
   }
   
   void MNISTParser::readTestLabels(string dir){
      basic_ifstream<unsigned char> file((dir + testLabelFile).c_str(),ios::in|ios::binary|ios::ate);
      testLabels = new MNISTTestLabels;
      
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTestLabels(" << (dir + testLabelFile) << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(testLabels->data,size);
      
      if(testLabels->formatted.magicNumber != 2049)
         cout << "csvm::MNISTParser::readTestLabels: Magic number mismatch! It says " << testLabels->formatted.magicNumber << "\n";
      
      file.close();
   }
   
   vector<Image> MNISTParser::convertTrainSetToImages(){
      vector<Image> formattedImages;
      
      if(trainImages == NULL && trainLabels != NULL){
         size_t nImages = trainImages->formatted.numberOfImages;
         formattedImages.reserve(nImages);
         size_t width = trainImages->formatted.numberOfRows;
         size_t height = trainImages->formatted.numberofColumns;
         
         for(size_t imIdx = 0; imIdx < nImages; ++imIdx){
            formattedImages.push_back(Image(width, height, CSVM_IMAGE_UCHAR_GREY));
            formattedImages[imIdx].setImageData(vector<unsigned char>(trainImages->formatted.images[imIdx].pixels, trainImages->formatted.images[imIdx].pixels + sizeof(trainImages->formatted.images[imIdx].pixels) / sizeof(trainImages->formatted.images[imIdx].pixels[0])));
            char label = trainLabels->formatted.labels[imIdx];
            formattedImages[imIdx].setLabel(string(&label));
            formattedImages[imIdx].setLabelId(trainLabels->formatted.labels[imIdx]);
         }
         
      }
      return formattedImages;
   }
   
   
   vector<Image> MNISTParser::convertTestSetToImages(){
      vector<Image> formattedImages;
      
      if(trainImages == NULL && trainLabels != NULL){
         size_t nImages = testImages->formatted.numberOfImages;
         formattedImages.reserve(nImages);
         size_t width = testImages->formatted.numberOfRows;
         size_t height = testImages->formatted.numberofColumns;
         
         for(size_t imIdx = 0; imIdx < nImages; ++imIdx){
            formattedImages.push_back(Image(width, height, CSVM_IMAGE_UCHAR_GREY));
            formattedImages[imIdx].setImageData(vector<unsigned char>(testImages->formatted.images[imIdx].pixels, testImages->formatted.images[imIdx].pixels + sizeof(testImages->formatted.images[imIdx].pixels) / sizeof(testImages->formatted.images[imIdx].pixels[0])));
            char label = testLabels->formatted.labels[imIdx];
            formattedImages[imIdx].setLabel(string(&label));
            formattedImages[imIdx].setLabelId(testLabels->formatted.labels[imIdx]);
         }
         
      }
      return formattedImages;
   }
   
   void MNISTParser::deleteUnformattedData(){
      delete trainImages;
      delete trainLabels;
      delete testImages;
      delete testLabels;
   }
   

