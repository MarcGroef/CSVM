import hashlib
import time
import random
import os

import SVM.SVM as SVM
from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.random_bandit import Random_bandit
from generators.pso_ucbv import PSO_ucbv
from generators.cma_gen_ucbv import CMAES_ucbv
from generators.cma_gen import CMAES_advanced as CMAES
from ParamOpt import PerformTest

testers = ['BanditTester']

class BanditTester(ParameterTester):
    file_location = os.path.dirname(os.path.abspath(__file__))
    program_location = os.path.dirname(file_location)
    SVM_location = os.path.join(program_location, "SVM")

    start_command = "../SVM/SVM.out %(filename)s"
    param_path = "../SVM/"
    param_names = ['alpha', 'Falpha', 'beta', 'COST', 'FCOST', 'D2', 'FD2', 'SVM_C', 'FSVM_C',  'Y_FRACT', 'Y_INIT', 'ALPHA_ITER', 'ALPHA_ITER_INIT', 'FALPHA_ITER', 'NR_REP1', 'NR_REP2',
                   'EPS', 'F_EPS', 'SIGMA', 'FSIGMA', 'INIT_ALPHA', 'INIT_ALPHA_1', 'INIT_ALPHA_2', 'FINIT_ALPHA', 'FALPHA_ADDSTART', 'EPOCHS', 'FLAYER_SIZE']
    parameters = {
                    'alpha':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.000005,
                                        "max": 0.3,
                                        "distribution": "gaussian",
                                        "value": (0.02, 0.02)},
                    'Falpha':          {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 0.3,
                                        "distribution": "gaussian",
                                        "value": (0.003, 0.01)},
                    'beta':            {"type": "float",
                                        "scaling": "log",
                                        "min": 0.5,
                                        "max": 1.0,
                                        "distribution": "gaussian",
                                        "value": (0.7, 0.2)},
                    'COST':            {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.01,
                                        "max": 4.0,
                                        "distribution": "gaussian",
                                        "value": (0.1, 0.1)},
                    'FCOST':           {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.0,
                                        "max": 4.0,
                                        "distribution": "gaussian",
                                        "value": (0.5, 0.5)},
                    'D2':              {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.0,
                                        "max": 32.0,
                                        "distribution": "uniform"},
                    'FD2':             {"type": "float",
                                        "scaling": "linear",
                                        "min": -10.5,
                                        "max": 4.0,
                                        "distribution": "uniform"},
                    'SVM_C':           {"type": "float",
                                        "scaling": "log",
                                        "min": 1.0,
                                        "max": 80000.0,
                                        "distribution": "gaussian",
                                        "value": (6000.0, 10000.0)},
                    'FSVM_C':          {"type": "float",
                                        "scaling": "log",
                                        "min": 0.5,
                                        "max": 200.0,
                                        "distribution": "gaussian",
                                        "value": (40.0, 10.0)},
                   'Y_FRACT':          {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.8,
                                        "max": 1.0,
                                        "distribution": "uniform"},
                    'Y_INIT':          {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.05,
                                        "max": 1.0,
                                        "distribution": "uniform"},
                    'ALPHA_ITER':      {"type": "int",
                                        "scaling": "linear",
                                        "min": 5.0,
                                        "max": 25.0},
                    'ALPHA_ITER_INIT': {"type": "int",
                                        "scaling": "linear",
                                        "min": 5.0,
                                        "max": 25.0},
                    'FALPHA_ITER':     {"type": "int",
                                        "scaling": "linear",
                                        "min": 5.0,
                                        "max": 25.0},
                    'NR_REP1':         {"type": "static",
                                        "value": 1000},
                    'NR_REP2':         {"type": "int",
                                        "scaling": "linear",
                                        "min": 5.0,
                                        "max": 70.0},
                    'EPS':             {"type": "float",
                                        "scaling": "log",
                                        "min": 0.001,
                                        "max": 0.5,
                                        "distribution": "uniform"},
                    'F_EPS':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.00001,
                                        "max": 2.0,
                                        "distribution": "uniform"},
                    'SIGMA':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.05,
                                        "max": 18.0,
                                        "distribution": "uniform"},
                    'FSIGMA':          {"type": "float",
                                        "scaling": "log",
                                        "min": 0.03,
                                        "max": 50.0,
                                        "distribution": "uniform"},
                    'INIT_ALPHA':      {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.1,
                                        "max": 0.3},
                    'INIT_ALPHA_1':    {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.1,
                                        "max": 0.3},
                    'INIT_ALPHA_2':    {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.1,
                                        "max": 0.2},
                    'FINIT_ALPHA':     {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.1,
                                        "max": 0.2},
                    'FALPHA_ADDSTART': {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.01,
                                        "max": 0.1},
                    'EPOCHS':          {"type": "int",
                                        "scaling": "linear",
                                        "min": 1,
                                        "max": 1},
                    'FLAYER_SIZE':     {"type": "int",
                                        "scaling": "linear",
                                        "min": 5,
                                        "max": 50}}

    config_file = \
"""
algorithm
alpha                   %(alpha).8f
Falpha                  %(Falpha).8f
beta                    %(beta).8f
COST                    %(COST).8f
FCOST                   %(FCOST).8f
D2                      %(D2).8f
FD2                     %(FD2).8f
SVM_C                   %(SVM_C).8f
FSVM_C                  %(FSVM_C).8f
Y_FRACT                 %(Y_FRACT).8f
Y_INIT                  %(Y_INIT).8f
ALPHA_ITER              %(ALPHA_ITER)d
ALPHA_ITER_INIT         %(ALPHA_ITER_INIT)d
FALPHA_ITER             %(FALPHA_ITER)d
NR_REP1                 1
NR_REP2                 %(NR_REP2)d
EPS                     %(EPS).8f
F_EPS                   %(F_EPS).8f
SIGMA                   %(SIGMA).8f
FSIGMA                  %(FSIGMA).8f
INIT_ALPHA              %(INIT_ALPHA).8f
INIT_ALPHA_1            %(INIT_ALPHA_1).8f
INIT_ALPHA_2            %(INIT_ALPHA_2).8f
FINIT_ALPHA             %(FINIT_ALPHA).8f
FALPHA_ADDSTART         %(FALPHA_ADDSTART).8f
EPOCHS                  %(EPOCHS)d
FLAYER_SIZE             %(FLAYER_SIZE)d
"""

    def run_algorithm(self):
        #print "file_location %s" % self.file_location
        #print "program location %s" % self.program_location
        #print "SVM location %s" % self.SVM_location
        os.chdir(self.SVM_location)
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
                self.result = float(os.popen("./SVM.out " + filename).read())
                if self.result < 0.00001:
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
    result = tst.set_options(gen, BanditTester, 8, 100000, processing_timeout = 1800) # number of threads and single function evaluations
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
