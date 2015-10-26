all: python_module

python_module: SVM/SVM.out SVM/_SVM.so

clean:
	rm -f *.pyc */*.pyc SVM/Tester_Parameters_*
	cd SVM ; $(MAKE) $(MFLAGS) clean

SVM/SVM.out:
	cd SVM ; $(MAKE) $(MFLAGS) SVM.out

SVM/_SVM.so:
	cd SVM ; $(MAKE) $(MFLAGS) _SVM.so

