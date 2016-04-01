#!/bin/bash
swig -c++ -python csvm.i
g++ -Wall -O2 -fPIC -I/usr/include/python2.7 -I../include -c ../src/*/*.cc  experiment.cc csvm_wrap.cxx -lm

g++ -Wall -shared *.o -o ../PSO/testers/_csvm.so
rm *.o
#rm *.cxx

mv csvm.py ../PSO/testers/csvm.py
cd ../PSO/testers/
#rm *.pyc
