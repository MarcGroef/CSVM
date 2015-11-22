#include <csvm/csvm_mnist_parser.h>
#include <stdint.h>

using namespace std;
using namespace csvm;

   
   void MNISTParser::readTrainImages(string filename){
      basic_ifstream<unsigned char> file(filename.c_str(),ios::in|ios::binary|ios::ate);
            
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTrainImages(" << filename << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(trainImages.data,size);
      file.close();
   }
   
   void MNISTParser::readTrainLabels(string filename){
      basic_ifstream<unsigned char> file(filename.c_str(),ios::in|ios::binary|ios::ate);
            
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTrainLabels(" << filename << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(trainLabels.data,size);
      file.close();
   }
   
   void MNISTParser::readTestImages(string filename){
      basic_ifstream<unsigned char> file(filename.c_str(),ios::in|ios::binary|ios::ate);
            
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTestImages(" << filename << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(testImages.data,size);
      file.close();
   }
   
   void MNISTParser::readTestLabels(string filename){
      basic_ifstream<unsigned char> file(filename.c_str(),ios::in|ios::binary|ios::ate);
            
      if(!file.is_open()){
         cout << "csvm::MNISTParser::readTestLabels(" << filename << "): Warning! Could not open file!\n";
         return;
      }
      
      int size = file.tellg();
            
      file.seekg(0,ios::beg);
      file.read(testLabels.data,size);
      file.close();
   }
   

