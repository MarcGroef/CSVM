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
       'SVM_C_Data',
       'HOG_cellSize',
       'HOG_cellStride',
       #'PatchSize',
       'PatchStride',
       #'sigmaClassicSimilarity',
       #'similaritySigma',
       #'SVM_Iterations',
       'learningRate']
    parameters = {  'learningRate':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 0.01,
                                        "distribution": "uniform",},
                     
                    #'sigmaClassicSimilarity':           {"type": "float",
                    #                    "scaling": "linear",
                    #                    "min": 0.001,
                    #                    "max": 4.0,
                    #                    "distribution": "uniform",},
                    #'similaritySigma': {"type": "float",
                    #                    "scaling": "linear",
                    #                    "min": 0.001,
                    #                    "max": 4.0,
                    #                    "distribution": "uniform",},
                    'HOG_cellSize': {"type": "int",
                                        "scaling": "linear",
                                        "min": 2,
                                        "max": 12,
                                        "distribution": "uniform",},
                    'HOG_cellStride': {"type": "int",
                                        "scaling": "linear",
                                        "min": 2,
                                        "max": 12,
                                        "distribution": "uniform",},
                    #'PatchSize': {"type": "int",
                    #                    "scaling": "linear",
                    #                    "min": 8,
                    #                    "max": 32,
                    #                    "distribution": "uniform",},
                    'PatchStride': {"type": "int",
                                        "scaling": "linear",
                                        "min": 1,
                                        "max": 12,
                                        "distribution": "uniform",},
                    #'SVM_Iterations': {"type": "int",
                    #                    "scaling": "log",
                    #                    "min": 1000,
                    #                    "max": 50000,
                    #                    "distribution": "uniform",},
                    'SVM_C_Data':      {"type": "int",
                                        "scaling": "log",
                                        "min": 1.0,
                                        "max": 100000000}}
    config_file = \
"""Dataset
method CIFAR10
nImages 1000


ClusterAnalyser
method RBM
nLayers 2
layerSizes 100
learningRate 0.1
nGibbsSteps 2

Codebook
method KMEANS
nClusters 60
SimilarityFunction SOFT_ASSIGNMENT
similaritySigma 0.0001

FeatureExtractor
method HOG
cellSize %(HOG_cellSize)d
cellStride %(HOG_cellStride)d
blockSize 8
padding None

ImageScanner
patchHeight 12
patchWidth 12
scanStride %(PatchStride)d
nRandomPatches 50

SVM
Type CONV
Kernel LINEAR
AlphaDataInit 0.0001
AlphaCentroidInit 0.0001
nIterations 2000
learningRate %(learningRate).7f
SVM_C_Data %(SVM_C_Data)d
SVM_C_Centroid 1
Cost 1
D2 1
sigmaClassicSimilarity 100
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
                self.result = float(csvm.run(filename, "testers/codebook.bin", "../../datasets/"))
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
    result = tst.set_options(gen, BanditTester, 4, 10000, processing_timeout = 66000) # number of threads and single function evaluations
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
