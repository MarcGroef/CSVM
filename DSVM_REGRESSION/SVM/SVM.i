%module SVM

%{
#define SWIG_FILE_WITH_INIT
#include "SVM.h"
%}
double runSVM(char const *file);
