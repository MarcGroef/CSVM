import random
from param_generator import ParameterGenerator

generators = ['RandomParameters']

class RandomParameters(ParameterGenerator):
    """
    The RandomParameters subclass of ParameterGenerator
    returns a random value for each of the specified parameters.
    """
    def get_next_parameter(self, name, value, settings, id=0):
        if settings['type'] == "static":
            return settings['value'];
        elif settings['type'] == "int":
            if settings['distribution'] == "uniform":
                return random.randint(settings['min'], settings['max'])
            elif settings['distribution'] == "gaussian":
                param = settings['value'];
                return round(random.gauss(param[0], param[1]))
        elif settings['type'] == "float":
            if settings['distribution'] == "uniform":
                return random.uniform(settings['min'], settings['max'])
            elif settings['distribution'] == "gaussian":
                param = settings['value'];
                return random.gauss(param[0], param[1])
        elif settings['type'] == "choice":
            return random.choice(settings['value'])
        return None
