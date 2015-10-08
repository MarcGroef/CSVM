CFLAGS=-Wall -O4 -fPIC -pthread
PYTHONINCLUDE=`python-config --cflags` `python-config --ldflags`
#CFLAGS=-c -Wall -g -fPIC

all: SVM.out _SVM.so

#				SVM
SVM.out: SVM.cpp param_letters.c utils.cpp SIMUL.c SVM.h 
		g++ $(CFLAGS) SVM.cpp param_letters.c utils.cpp SIMUL.c -c
		g++ -o SVM.out SVM.o param_letters.o utils.o SIMUL.o
		@echo done compiling SVM

_SVM.so: SVM.out SVM_wrap.cxx SVM.py
	g++ $(CFLAGS) -shared -lstdc++ SVM.o param_letters.o utils.o SIMUL.o SVM_wrap.cxx $(PYTHONINCLUDE) -o _SVM.so

SVM.py: SVM.i
	swig -python -c++ SVM.i

SVM_wrap.cxx: SVM.i
	swig -python -c++ SVM.i

clean:
	rm -f *.out *.o *.so SVM_wrap.cxx SVM.py *.pyc 
	@echo done removing

