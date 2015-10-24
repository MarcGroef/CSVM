#!/bin/bash
swig -c++ -python csvm.i
g++ -O3 -fPIC -I/usr/include/python2.7 -I../include -c ../src/*/*.cc  experiment.cc csvm_wrap.cxx -lm

g++ -shared *.o -o _csvm.so
rm *.o