#include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;

void CSVMSettings::readSettingsFile(string dir){
   ifstream file(dir.c_str(),ios::in);
   string line;
   bool seenAlgorithm = false;
   
   if(!file.is_open()){
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }
   
   while(getline(file,line)){
      if(!seenAlgorithm && line == "algorithm")
         seenAlgorithm;
      else{
          cout << line << "\n";
      }
   }
   
}