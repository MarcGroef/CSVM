import hashlib
import time
import random
import os

import SVM.SVM as SVM
from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.xnes_bare import xNES_bare as xNES
from generators.pso_threaded import PSO
from ParamOpt import PerformTest

testers = ['NumperTester']

class NumperTester(ParameterTester):
    file_location = os.path.dirname(os.path.abspath(__file__))
    program_location = os.path.dirname(file_location)
    SVM_location = os.path.join(program_location, "SVM")

    start_command = "SVM/SVM.out %(filename)s"
    param_path = "SVM"
    param_names = ['discreteActions', 'stateDimension', 'actionDimension', 'alpha',
                   'beta', 'COST', 'D2', 'SVM_C', 'ALPHA_ITER', 'NR_REP1', 'NR_REP2',
                   'EPS', 'nHiddenV']
    parameters = {
                    'discreteActions': {"type": "static",
                                        "value": 1},
                    'stateDimension':  {"type": "static",
                                        "value": 6},
                    'actionDimension': {"type": "int",
                                        "scaling": "log",
                                        "min": 1,
                                        "max": 100},
                    'alpha':           {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 0.2,
                                        "distribution": "gaussian",
                                        "value": (0.002, 0.001)},
                    'beta':            {"type": "float",
                                        "scaling": "log",
                                        "min": 0.0000001,
                                        "max": 0.1,
                                        "distribution": "gaussian",
                                        "value": (0.006, 0.001)},
                    'COST':            {"type": "float",
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
                    'SVM_C':           {"type": "int",
                                        "scaling": "linear",
                                        "min": 4.0,
                                        "max": 16.0},
                    'ALPHA_ITER':      {"type": "int",
                                        "scaling": "linear",
                                        "min": 1.0,
                                        "max": 10.0},
                    'NR_REP1':         {"type": "static",
                                        "value": 10},
                    'NR_REP2':         {"type": "static",
                                        "value": 10},
                    'EPS':             {"type": "float",
                                        "scaling": "log",
                                        "min": 0.001,
                                        "max": 0.1,
                                        "distribution": "uniform"},
                    'nHiddenV':        {"type": "int",
                                        "scaling": "linear",
                                        "min": 1.0,
                                        "max": 30}}
    config_file = \
"""world
discreteActions         %(discreteActions)d
stateDimension          %(stateDimension)d
actionDimension         %(actionDimension)d

algorithm
alpha                   %(alpha).7f
beta                    %(beta).7f
COST                    %(COST)f
D2                      %(D2)f
SVM_C                   %(SVM_C)f
ALPHA_ITER              %(ALPHA_ITER)d
NR_REP1                 %(NR_REP1)d
NR_REP2                 %(NR_REP2)d
EPS                     %(EPS)f

nn
nHiddenV                %(nHiddenV)d
"""

    def run_algorithm(self):
        print "file_location %s" % self.file_location
        print "program location %s" % self.program_location
        print "SVM location %s" % self.SVM_location
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
                    if tryCount >= 5:
                        print "[NOTICE] Unrealistically low result: %.8f for parameters %s after 5 tries. Continuing." % (self.result, repr(self.parameters))
                        break
                    else:
                        print "[NOTICE] Unrealistically low result: %.8f for parameters %s. Restarting." % (self.result, repr(self.parameters))
                    continue
                break
        finally:
            os.unlink(filename) # Make sure the file is always deleted

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
