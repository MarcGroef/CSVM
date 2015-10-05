 #include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;


void CSVMSettings::parseDatasetSettings(ifstream& stream){

  
  string setting;
  string method;
  stream >> setting;
  if(setting != "method"){
    cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Crashing...\n";
    exit(-1);
  }
  stream >> method;

  if(method == "CIFAR10"){
      stream >> setting;
      if(setting != "nImages"){
	cout << "csvm::csvm_settings:parseDatasetSettings(): In CIFAR10 parsing: Error! Invalid settingsfile layout. Crashing...\n";
	exit(-1);
      }
      stream >> datasetSettings.nImages;
    
  }
 
}



void CSVMSettings::readSettingsFile(string dir){
   ifstream file(dir.c_str(),ios::in);
   string line;

   
   if(!file.is_open()){
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }
   
   while(getline(file,line) && line != "Dataset");
   parseDatasetSettings(file);
   while(getline(file,line) && line != "ClusterAnalyser");
   
   while(getline(file,line) && line != "Codebook");
   
   while(getline(file,line) && line != "FeatureExtractor");
   
   while(getline(file,line) && line != "ImageScanner");
   // parse values:
   
   /*string temp;
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
   file >> temp >> svmSettings.INIT_ALPHA;*/
   
   
   file.close();   
}