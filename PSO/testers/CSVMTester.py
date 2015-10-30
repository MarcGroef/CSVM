import hashlib
import time
import random
import os

from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.xnes_bare import xNES_bare as xNES
from generators.pso_threaded import PSO
from ParamOpt import PerformTest

import csvm

testers = ['NumperTester']

class NumperTester(ParameterTester):
    #file_location = os.path.dirname(os.path.abspath(__file__))
    #program_location = os.path.dirname(file_location)
    #SVM_location = os.path.join(program_location, "../build/CSVM")
    
    start_command = "."
    param_path = "."
    param_names = [ 'SVM_C_Data', 'sigmaClassicSimilarity', 'similaritySigma']
    parameters = {
                    'sigmaClassicSimilarity':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 2.0,
                                        "distribution": "uniform",},
                    'similaritySigma': {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 2.0,
                                        "distribution": "uniform",},
                    
                    'SVM_C_Data':      {"type": "int",
                                        "scaling": "linear",
                                        "min": 1.0,
                                        "max": 16.0}}
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
nClusters 300
similaritySigma %(similaritySigma).7f

FeatureExtractor
method HOG
cellSize 4
cellStride 2
blockSize 8

ImageScanner
patchHeight 8
patchWidth 8
scanStride 1
nRandomPatches 100

SVM
learningRate 0.00001
SVM_C_Data %(SVM_C_Data).7f
SVM_C_Centroid 1
Cost 1
D2 1
sigmaClassicSimilarity %(sigmaClassicSimilarity).7f
"""

    def run_algorithm(self):
        #print "file_location %s" % self.file_location
        #print "program location %s" % self.program_location
        #print "CSVM location %s" % self.SVM_location
        #os.chdir(self.SVM_location)
        #file_location = os.path.dirname(os.path.abspath(__file__))
        #program_location = os.path.dirname(file_location)
        #os.chdir(program_location)
        
        dig = hashlib.md5()
        dig.update(str(time.time()))
        dig.update(str(time.time()))
        dig.update(str(random.random()) + str(random.random()))
        filename = "Tester_Parameters_%s" % dig.hexdigest()
        self.write_parameters(filename)
        try:
            #tryCount = 0
            #while True:
                #tryCount += 1
                #print "Calling : ./CSVM " + filename
                #process = os.popen("../build/CSVM " + filename)
                #answer = process.read()
                #process.kill()
            self.result = float(csvm.run(filename, "testers/codebook.bin", "../datasets/"))
            print "Reaction from program = ", self.result
                #self.result = float(answer)
                #if self.result < -0.00001:
                #    if tryCount >= 5:
                ##        break
                #    else:
                #        print "[NOTICE] Unrealistically low result: %.8f for parameters %s. Restarting." % (self.result, repr(self.parameters))
                #    continue
                #break
        finally:
            os.unlink(filename) # Make sure the file is always deleted
        #return self.result
        

       
if __name__ == "__main__":
    #gen = PSO()
    #gen = CMAES()
    gen = xNES()
    NumperTester.add_parameters(gen)

    tst = PerformTest()
    result = tst.set_options(gen, NumperTester, 1, 5000, processing_timeout=300)
    if not result is True:
        print result
        tst.stop_running()
        exit()

    tst.start_evaluation()
    try:
        while tst.is_alive():
            time.sleep(0.01)
    except KeyboardInterrupt:
        tst.stop_running()
        raise
