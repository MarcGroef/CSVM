import hashlib
import time
import random
import os
import csvm

from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.cma_gen import CMAES_advanced as CMAES
from ParamOpt import PerformTest

testers = ['GenericTester']

class GenericTester(ParameterTester):

    start_command = ""
    param_path = "CSVM"
    param_names = [ 'Cost', 'D2', 'SVM_C_Data', 'SVM_C_Centroid', 'sigmaClassicSimilarity', 'similaritySigma']
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
                    'Cost':            {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.0,
                                        "max": 4.0,
                                        "distribution": "gaussian",
                                        "value": (0.5, 0.25)},
                    'D2':              {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.0,
                                        "max": 4.0,
                                        "distribution": "uniform"},
                    'SVM_C_Centroid':  {"type": "int",
                                        "scaling": "linear",
                                        "min": 4.0,
                                        "max": 16.0},
                    'SVM_C_Data':      {"type": "int",
                                        "scaling": "linear",
                                        "min": 4.0,
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
cellSize 8
cellStride 1
blockSize 16

ImageScanner
patchHeight 8
patchWidth 8
scanStride 1
nRandomPatches 100

SVM
learningRate 0.00001
SVM_C_Data %(SVM_C_Data).7f
SVM_C_Centroid %(SVM_C_Centroid).7f
Cost %(Cost).7f
D2 %(D2).7f
sigmaClassicSimilarity %(sigmaClassicSimilarity).7f
"""

    def run_algorithm(self):
        file_location = os.path.dirname(os.path.abspath(__file__))
        program_location = os.path.dirname(file_location)
        os.chdir(program_location)

        dig = hashlib.md5()
        dig.update(str(time.time()))
        dig.update(str(time.time()))
        dig.update(str(random.random()) + str(random.random()))

        param_path = os.path.abspath(self.param_path)
        filename = "Test_Parameters_%s" % dig.hexdigest()
        if param_path[-1:] != "/":
            filename = "/" + filename
        filepath = param_path + filename
        self.write_parameters(filepath)

        working_dir = os.path.dirname(os.path.abspath(self.start_command))
        start_cmd = os.path.basename(self.start_command)
        os.chdir(working_dir)

        if start_cmd.find("%(filename)s") == -1:
            start_cmd += " %(filename)s"

        command = start_cmd % {'filename': filepath}
        command = "./" + command

        try:
            self.result = float(os.popen(command).read())
        finally:
            os.unlink(filepath) # Make sure the file is always deleted
