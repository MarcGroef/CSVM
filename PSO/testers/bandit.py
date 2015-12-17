import hashlib
import time
import random
import os

import csvm

from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.random_bandit import Random_bandit
from generators.pso_ucbv import PSO_ucbv
from generators.cma_gen_ucbv import CMAES_ucbv
from generators.cma_gen import CMAES_advanced as CMAES
from ParamOpt import PerformTest

testers = ['BanditTester']

class BanditTester(ParameterTester):
    #file_location = os.path.dirname(os.path.abspath(__file__))
    #program_location = os.path.dirname(file_location)
    #SVM_location = os.path.join(program_location, "../build/CSVM")
    
    start_command = "."
    param_path = "."
    param_names = [
       #'nCentroids',
       #'patchSize',
       #'patchStride',
       'learningRate',
       #'nTrainingIterations',
       #'codebookSimilaritySigma',
       #'SVMSimilaritySigma',
       'SVM_C']
    parameters = {  'learningRate':   {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 0.01,
                                        "distribution": "uniform",},
    
                    #'nCentroids':       {"type": "int",
                    #                    "scaling": "linear",
                    #                    "min": 100,
                    #                    "max": 1000,
                    #                    "distribution": "uniform",},

                    #'codebookSimilaritySigma': {"type": "float",
                    #                    "scaling": "log",
                    #                    "min": 0.0001,
                    #                    "max": 100.0,
                    #                    "distribution": "uniform",},
                    #'SVMSimilaritySigma': {"type": "float",
                    #                    "scaling": "log",
                    #                    "min": 0.0001,
                    #                    "max": 100.0,
                    #                    "distribution": "uniform",},
                    #'patchSize': {"type": "int",
                    #                    "scaling": "linear",
                    #                    "min": 8,
                    #                    "max": 32,
                    #                    "distribution": "uniform",},
                    #'patchStride': {"type": "int",
                    #                    "scaling": "linear",
                    #                    "min": 1,
                    #                    "max": 8,
                    #                    "distribution": "uniform",},
                    #'nTrainingIterations': {"type": "int",
                    #                    "scaling": "log",
                    #                    "min": 1000,
                    #                    "max": 50000,
                    #                    "distribution": "uniform",},
       
                    'SVM_C':            {"type": "int",
                                        "scaling": "log",
                                        "min": 1.0,
                                        "max": 1000000}}
    config_file = \
"""
Dataset
method MNIST
nTrainImages 60000
nTestImages 10000

General
Classifier CSVM
Codebook CODEBOOK
nClasses 10

Codebook
method KMEANS
nClusters 500
nIterations 20
SimilarityFunction SOFT_ASSIGNMENT
similaritySigma 0.2


FeatureExtractor
method HOG
cellSize 6
cellStride 6
blockSize 8
padding None
useGreyPixel true

ImageScanner
patchHeight 12
patchWidth 12
scanStride 2
nRandomPatches 20000

SVM
Kernel LINEAR
AlphaDataInit 0.001
nIterations 1000
learningRate %(learningRate).7f
SVM_C_Data %(SVM_C)d
Cost 1
D2 1
sigmaClassicSimilarity 0.002

LinNet
nIterations 1000
initWeight 0.01
learningRate %(learningRate).7f

ConvSVM
learningRate %(learningRate).7f
nIterations 50000
initWeight 0.002
CSVM_C %(SVM_C).7f
"""

    def run_algorithm(self):
        #print "file_location %s" % self.file_location
        #print "program location %s" % self.program_location
        #print "SVM location %s" % self.SVM_location
        #os.chdir(self.SVM_location)
        dig = hashlib.md5()
        dig.update(str(time.time()))
        dig.update(str(time.time()))
        dig.update(str(random.random()) + str(random.random()))
        filename = "Tester_Parameters_%s" % dig.hexdigest()
        self.write_parameters(filename)
        try:
            tryCount = 0
            while True:
                tryCount += 1
                #self.result = float(os.popen("./SVM.out " + filename).read())
                self.result = float(csvm.run(filename, "codebook50000HOG.bin", "../../datasets/"))
                if self.result < -0.00001:
                    if tryCount < 5:
                        print "[NOTICE] Unrealistically low result: %.8f for parameters %s. Restarting." % (self.result, repr(self.parameters))
                        continue
                    else:
                        print "[NOTICE] Unrealistically low result: %.8f for parameters %s after 5 tries. Continuing." % (self.result, repr(self.parameters))
                break
        finally:
            os.unlink(filename) # Make sure the file is always deleted

if __name__ == "__main__":
    gen = PSO_ucbv()
    #gen = Random_bandit()
    #gen = CMAES()
    BanditTester.add_parameters(gen)

    tst = PerformTest()
    result = tst.set_options(gen, BanditTester, 6, 10, processing_timeout = 66000) # number of threads and single function evaluations

    if not result is True:
        print result
        tst.stop_running()
        exit()

    tst.start_evaluation()
    try:
        while tst.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        tst.stop_running()
        raise
