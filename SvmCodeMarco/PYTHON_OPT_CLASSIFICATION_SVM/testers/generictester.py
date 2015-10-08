import hashlib
import time
import random
import os

from param_tester import ParameterTester
from generators.randomparameters import RandomParameters
from generators.cma_gen import CMAES_advanced as CMAES
from ParamOpt import PerformTest

testers = ['GenericTester']

class GenericTester(ParameterTester):

    start_command = ""
    param_path = ""
    param_names = ['discreteActions', 'stateDimension', 'actionDimension', 'alpha',
                   'beta', 'COST', 'D2', 'SVM_C', 'ALPHA_ITER', 'NR_REP1', 'NR_REP2',
                   'EPS', 'nHiddenV']
    parameters = {
                    'discreteActions': {"type": "static",
                                        "value": 1},
                    'stateDimension':  {"type": "static",
                                        "value": 2},
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
                                        "value": 100},
                    'NR_REP2':         {"type": "static",
                                        "value": 50},
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
COST                    %(COST).7f
D2                      %(D2).7f
SVM_C                   %(SVM_C).7f
ALPHA_ITER              %(ALPHA_ITER)d
NR_REP1                 %(NR_REP1)d
NR_REP2                 %(NR_REP2)d
EPS                     %(EPS).7f

nn
nHiddenV                %(nHiddenV)d
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
