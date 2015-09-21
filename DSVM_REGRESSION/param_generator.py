import abc

############################################
## ParameterGenerator Abstract Base Class ##
############################################
class ParameterGenerator(object):
    """
    The ParameterGenerator is an abstract base class that can be
    used to generate new parameters for an parameter evaluation
    mechanism. It is used to set up new parameters and then generate
    new values based on the specifications of the parameter and
    previously generated values
    """
    __metaclass__ = abc.ABCMeta

    ####################
    ## Common Methods ##
    ####################
    def __init__(self):
        """The constructor initializes empty lists"""
        self.param_names = []
        self.param_values = []
        self.param_settings = []
        self.result = []
        self.best_params = None
        self.best_score = None
        self.max_reps = 200
        self.num_values = False
        self.algorithm_done = False

    def set_num_values(self, num_values):
        self.num_values = num_values

    def generate_parameters(self):
        """
        This method generates a new value for each parameter
        by calling the generate_parameter function for each
        defined parameter, passing along its settings.
        """
        id = len(self.param_values)

        vector = []
         
        for idx in range(len(self.param_names)):
            name = self.param_names[idx]
            value = None
            settings = self.param_settings[idx]
            new_value = self.get_next_parameter(name, value, settings, id=id)
            vector.append(self.clamp(new_value, settings['min'], settings['max']))

        self.param_values.append(vector)
        self.result.append(None)
        return id, vector

    def set_result(self, params, result, id=0):
        """
        This method sets the result/evalution of the last set of
        parameters. This can be used by subclasses to generate the next
        value for each parameter.
        """
        for i in range(len(self.result), id + 1):
            self.result.append(None)
            print "List index %d does not exist, should not happen" % i
        self.result[id] = result
        if self.best_score is None or result < self.best_score:
            self.best_score = result
            self.best_params = params
        return params, result

    def get_best_result(self):
        return self.best_params, self.best_score

    def get_parameters(self):
        """
        This method generates new parameters and returns a dictionary
        with the new value for each parameter.
        """
        result = self.generate_parameters()
        id, vector = result
        if not vector:
            return id, vector

        params = {}
        for idx in range(len(self.param_names)):
            name = self.param_names[idx]
            settings = self.param_settings[idx]
            value = vector[idx]
            if settings['type'] == "int":
                value = int(round(value))
            elif settings['type'] == "float":
                value = float(value)
            params[name] = value
        return id, params

    def add_parameter(self,
                      name,           # The name of the parameter
                      scaling=None,   # The type of scaling to be used for the parameter
                      type="int",     # The type of the parameter, such as float
                      min=0,          # The minimum value of the parameter
                      max=100,        # The maximum value of the parameter
                      significance=1, # The smallest significant step size
                      value=None,     # The value or value parameters
                      distribution=None): # The distribution of the parameter
        """
        This method defines a new parameter, specifying some parameters
        that set the significance, minimum and maximum other relevant
        settings for the parameter.
        """
        config = {"scaling" : scaling, 
                  "type": type,
                  "min": min, 
                  "max": max, 
                  "significance": significance,
                  "value": value,
                  "distribution": distribution}
        self.param_names.append(name)
        self.param_settings.append(config)

    def set_max_reps(self, max_reps):
        """
        Sets the maximum number of evaluations of a single set of parameters, if
        the algorithm needs it
        """
        self.max_reps = int(max_reps)

    def present_result(self, parameters, result):
        """Present the results in some way. Could be overridden"""
        print "Result for parameters %s: %.7f" % (repr(parameters), result)

    def clamp(self, value, minVal, maxVal):
        """Clamp a numeric value between a minimum and a maximum"""
        if type(value) is type("string"):
            return value
        if minVal != None and max != None:
            return max(min(value, maxVal), minVal)
        if minVal != None and maxVal == None:
            return max(value, minVal)
        if minVal == None and maxVal != None:
            return min(value, maxVal)
        return value

    ####################
    ## Abstract Method #
    ####################
    @abc.abstractmethod
    def get_next_parameter(self, name, value, settings, id=0):
        """
        This method should return the new value for the parameter,
        based on its settings. This method should be overridden by
        subclasses to perform some sort of parameter generation
        """
        return None


