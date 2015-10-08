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
    param_names = ['alpha', 'beta', 'COST', 'D2', 'SVM_C', 'ALPHA_ITER', 'NR_REP1', 'NR_REP2',
                   'EPS', 'SIGMA', 'INIT_ALPHA']
    parameters = {
                    'alpha':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.00003,
                                        "max": 0.1,
                                        "distribution": "gaussian",
                                        "value": (0.005, 0.0005)},
                    'beta':            {"type": "float",
                                        "scaling": "log",
                                        "min": 0.2,
                                        "max": 0.2,
                                        "distribution": "gaussian",
                                        "value": (0.03, 0.01)},
                    'COST':            {"type": "float",
                                        "scaling": "linear",
                                        "min": -1.0,
                                        "max": 16.0,
                                        "distribution": "gaussian",
                                        "value": (2.0, 1.0)},
                    'D2':              {"type": "float",
                                        "scaling": "linear",
                                        "min": -1.0,
                                        "max": 10.0,
                                        "distribution": "uniform"},
                    'SVM_C':           {"type": "float",
                                        "scaling": "linear",
                                        "min": 1.0,
                                        "max": 10048.0},
                    'ALPHA_ITER':      {"type": "int",
                                        "scaling": "linear",
                                        "min": 5.0,
                                        "max": 45.0},
                    'NR_REP1':         {"type": "static",
                                        "value": 1000},
                    'NR_REP2':           {"type": "int",
                                        "scaling": "linear",
                                        "min": 1.0,
                                        "max": 60.0},
                    'EPS':             {"type": "float",
                                        "scaling": "log",
                                        "min": 0.1,
                                        "max": 0.1,
                                        "distribution": "uniform"},
                    'SIGMA':             {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.05,
                                        "max": 10.0,
                                        "distribution": "uniform"},
                    'INIT_ALPHA':        {"type": "float",
                                        "scaling": "linear",
                                        "min": 0.3,
                                        "max": 0.3}}

    config_file = \
"""
algorithm
alpha                   %(alpha).8f
beta                    %(beta).8f
COST                    %(COST).8f
D2                      %(D2).8f
SVM_C                   %(SVM_C).8f
ALPHA_ITER              %(ALPHA_ITER)d
NR_REP1                 1
NR_REP2                 %(NR_REP2)d
EPS                     %(EPS).8f
SIGMA                   %(SIGMA).8f
INIT_ALPHA              %(INIT_ALPHA).8f
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
    result = tst.set_options(gen, BanditTester, 8, 100000, processing_timeout = 6600) # number of threads and single function evaluations
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
