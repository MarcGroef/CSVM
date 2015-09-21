OBJECTS=  Matrix.o SVM.o  utils.o
CXXFLAGS= -O4 -fPIC -pthread
PYTHONINCLUDE=`python-config --cflags` `python-config --ldflags`
#CFLAGS=-c -Wall -g -fPIC

all: SVM.out _SVM.so

#				SVM
SVM.out: $(OBJECTS) SVM.h 
		g++ $(CXXFLAGS) param_letters.c SIMUL.c -c
		g++ -o SVM.out $(OBJECTS) SIMUL.o param_letters.o
		@echo done compiling SVM

_SVM.so: SVM.out SVM_wrap.cxx SVM.py
	g++ $(CXXFLAGS) -shared -lstdc++ $(OBJECTS) param_letters.o SIMUL.o SVM_wrap.cxx $(PYTHONINCLUDE) -o _SVM.so

SVM.py: SVM.i
	swig -python -c++ SVM.i

SVM_wrap.cxx: SVM.i
	swig -python -c++ SVM.i

clean:
	rm -f *.out *.o *.so SVM_wrap.cxx SVM.py *.pyc 
	@echo done removing

