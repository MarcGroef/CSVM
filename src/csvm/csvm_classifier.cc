#include <csvm/csvm_classifier.h>

using namespace std;
using namespace csvm;

void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
}

void constructCodebook(){
   
   
}