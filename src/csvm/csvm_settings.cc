 #include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;


void CSVMSettings::readSettingsFile(string dir){
   ifstream file(dir.c_str(),ios::in);
   string line;

   
   if(!file.is_open()){
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }
   
   while(getline(file,line) && line != "algorithm");
   
   // parse values:
   
   string temp;
   file >> temp >> svmSettings.alpha;
   file >> temp >> svmSettings.beta;
   file >> temp >> svmSettings.COST;
   file >> temp >> svmSettings.D2;
   file >> temp >> svmSettings.SVM_C;
   file >> temp >> svmSettings.ALPHA_ITER;
   file >> temp >> svmSettings.NR_REP1;
   file >> temp >> svmSettings.NR_REP2;
   file >> temp >> svmSettings.EPS;
   file >> temp >> svmSettings.SIGMA;
   file >> temp >> svmSettings.INIT_ALPHA;
   
   
   file.close();   
}