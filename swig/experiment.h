#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <csvm/csvm.h>
//#include <dnn/dnn.h>
#include <iostream>
#include <time.h>
#include <cstdlib>

void generateCodebook(char* settingsDir, char* codebook, char* dataDir);
double run(char* settingsDir, char* codebook, char* dataDir);
void help();
#endif
