import wx
import os
import string
import md5
from multiprocessing import cpu_count

from performtest import PerformTest
import ParamGUI
import cPickle

myEVT_PARAM_UPDATE = wx.NewEventType()
EVT_PARAM_UPDATE = wx.PyEventBinder(myEVT_PARAM_UPDATE, 1)
class UpdateEvent(wx.PyCommandEvent):
    """Event to signal that there is some text to update"""
    def __init__(self, etype, eid, value=None):
        """Creates the event object"""
        wx.PyCommandEvent.__init__(self, etype, eid)
        self._value = value

    def get_value(self):
        """Returns the value from the event.
        @return: the value of this event

        """
        return self._value

class ParameterOptimizerGUI(wx.App):
    def __init__(self):
        super(ParameterOptimizerGUI, self).__init__()
        self.modules = {}
        self.generators = []
        self.testers = []
        self._results = []
        self._result_labels = []
        self._hbox_results = None
        self._hbox_result_labels = None
        self._hbox_results_2 = None
        self._hbox_result_labels_2 = None
        self.threads = cpu_count()
        self.num_values = 1000
        self.buffer_limit = 8000
        self.test_performer = PerformTest()
        self.setup_gui()
        self.load_config()
        self.window.CenterOnScreen()

    ########################
    ## GUI INITIALIZATION ##
    ########################
    def setup_gui(self):
        self.window = wx.Frame(None, title="Parameter Optimizer", size=(950, 500))
        self.background = wx.Panel(self.window)
        
        self.generatorLabel = wx.StaticText(self.background, label='Parameter Generators')
        self.generatorList = wx.ComboBox(self.background, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.testerLabel = wx.StaticText(self.background, label='Parameter Testers')
        self.testerList = wx.ComboBox(self.background, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.threadLabel = wx.StaticText(self.background, label='Number of threads')
        self.threadList = wx.ComboBox(self.background, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.threadList.Bind(wx.EVT_COMBOBOX, self.update_threads)
        self.startButton = wx.Button(self.background, label='Start Testing')
        self.startButton.Bind(wx.EVT_BUTTON, self.start_test)
        self.stopButton = wx.Button(self.background, label='Stop Testing')
        self.stopButton.Enable(False)
        self.stopButton.Bind(wx.EVT_BUTTON, self.stop_test)
        self.configButton = wx.Button(self.background, label='Configure')
        self.configButton.Bind(wx.EVT_BUTTON, self.on_config)
        self.valueLabel = wx.StaticText(self.background, label='# Evaluations')
        self.valueList = wx.ComboBox(self.background, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.valueList.Bind(wx.EVT_COMBOBOX, self.update_values)
        self.timeoutLabel = wx.StaticText(self.background, label='Timeout (s)')
        self.timeout = wx.TextCtrl(self.background, value="30")
        self.bestScoreLabel = wx.StaticText(self.background, label='Best score:')
        self.bestScore = wx.TextCtrl(self.background, style=wx.TE_READONLY, value="N/A")
        self.bestParamsLabel = wx.StaticText(self.background, label='Best parameters:')
        self.bestParams = wx.TextCtrl(self.background, style=wx.TE_MULTILINE | wx.HSCROLL | wx.TE_READONLY)
        self.progressLabel = wx.StaticText(self.background, label='Progress:')
        self.progress = wx.Gauge(self.background, style=wx.GA_SMOOTH | wx.GA_HORIZONTAL)
        self.progress.SetRange(100)
        self.progress.SetValue(0)
        font = self.bestParams.GetFont()
        font = wx.Font(font.GetPointSize(), wx.TELETYPE,
                       font.GetStyle(),
                       font.GetWeight(), font.GetUnderlined())
        self.bestParams.SetFont(font)   

        # Fill the comboboxes
        self.find_generators()
        items = self.generatorList.GetItems()
        if items:
            self.generatorList.SetStringSelection(items[0])

        self.find_testers()
        items = self.testerList.GetItems()
        if items:
            self.testerList.SetStringSelection(items[0])

        for i in range(1, cpu_count() + 1):
            self.threadList.Append(str(i))
        self.threadList.SetStringSelection(str(self.threads))

        for i in range(0, 25):
            if i == 0:
                continue
            self.valueList.Append(str(i))
        for i in range(25, 100, 5):
            self.valueList.Append(str(i))
        for i in range(100, 1000, 25):
            self.valueList.Append(str(i))
        for i in range(1000, 10000, 1000):
            self.valueList.Append(str(i))
        for i in range(10000, 100000, 5000):
            self.valueList.Append(str(i))
        for i in range(100000, 1000001, 25000):
            self.valueList.Append(str(i))
        self.valueList.SetStringSelection(str(self.num_values))

        hboxLabels = wx.BoxSizer()
        hboxLabels.Add(self.generatorLabel, proportion=3, flag=wx.EXPAND)
        hboxLabels.Add(self.testerLabel, proportion=3, flag=wx.EXPAND)
        hboxLabels.Add(self.threadLabel, proportion=1, flag=wx.EXPAND)

        hboxCombos = wx.BoxSizer()
        hboxCombos.Add(self.generatorList, proportion=3, flag=wx.EXPAND)
        hboxCombos.Add(self.testerList, proportion=3, flag=wx.EXPAND | wx.LEFT, border=5)
        hboxCombos.Add(self.threadList, proportion=1, flag=wx.EXPAND | wx.LEFT, border=5)
        
        hboxButtons = wx.BoxSizer()
        hboxButtons.Add(self.startButton, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.stopButton, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.configButton, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.valueLabel, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.valueList, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.timeoutLabel, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.timeout, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.bestScoreLabel, proportion=0, flag=wx.ALL, border=5)
        hboxButtons.Add(self.bestScore, proportion=0, flag=wx.ALL, border=5)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(hboxLabels, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(hboxCombos, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(hboxButtons, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(self.bestParamsLabel, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(self.bestParams, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(self.progressLabel, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(self.progress, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.background.SetSizer(self.vbox, True)
        self.setup_result_boxes(self.threads)

        self.window.Bind(wx.EVT_CLOSE, self.on_close)
        self.window.Show()
        self.Bind(EVT_PARAM_UPDATE, self.on_param_update)

    def setup_result_boxes(self, threads):
        oldThreads = len(self._result_labels)
        for idx in range(oldThreads):
            if oldThreads <= 4 or idx < (oldThreads / 2):
                self._hbox_result_labels.Remove(self._result_labels[idx])
                self._hbox_results.Remove(self._results[idx])
            else:
                self._hbox_result_labels_2.Remove(self._result_labels[idx])
                self._hbox_results_2.Remove(self._results[idx])
            self._result_labels[idx].Destroy()
            self._results[idx].Destroy()

        self._result_labels = []
        self._results = []

        if self._hbox_result_labels:
            self.vbox.Remove(self._hbox_result_labels)
        if self._hbox_results:
            self.vbox.Remove(self._hbox_results)
        if self._hbox_result_labels_2:
            self.vbox.Remove(self._hbox_result_labels_2)
        if self._hbox_results_2:
            self.vbox.Remove(self._hbox_results_2)

        self._hbox_result_labels = wx.BoxSizer()
        self._hbox_results = wx.BoxSizer()
        if threads > 4:
            self._hbox_result_labels_2 = wx.BoxSizer()
            self._hbox_results_2 = wx.BoxSizer()
        else:
            self._hbox_result_labels_2 = None
            self._hbox_results_2 = None
            
        for i in range(threads):
            label = wx.StaticText(self.background, label='Results Thread %d' % (i + 1))
            text = wx.TextCtrl(self.background, style=wx.TE_MULTILINE | wx.HSCROLL | wx.TE_READONLY)
            self._result_labels.append(label)
            self._results.append(text)
            if threads <= 4 or i < (threads / 2):
                self._hbox_result_labels.Add(self._result_labels[i], proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
                self._hbox_results.Add(self._results[i], proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
            else:
                self._hbox_result_labels_2.Add(self._result_labels[i], proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
                self._hbox_results_2.Add(self._results[i], proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.vbox.Add(self._hbox_result_labels, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vbox.Add(self._hbox_results, proportion=3, flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=5)
        if threads > 4:
            self.vbox.Add(self._hbox_result_labels_2, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
            self.vbox.Add(self._hbox_results_2, proportion=3, flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=5)
        self.background.GetSizer().Layout()

    ####################
    ## EVENT HANDLERS ##
    ####################
    def on_close(self, event):
        self.write_config()
        if self.test_performer:
            self.test_performer.stop_running()
        self.window.Destroy()
    
    def update_threads(self, event):
        val = self.threadList.GetValue()
        self.threads = int(val)
        self.setup_result_boxes(self.threads)

    def update_values(self, event):
        val = self.valueList.GetValue()
        self.num_values = int(val)

    def start_test(self, event):
        generatorClass = self.extract_class(self.generatorList)
        testerClass = self.extract_class(self.testerList)
        generator = generatorClass()

        testerClass.add_parameters(generator)
        try:
            timeout = int(self.timeout.GetValue())
        except:
            timeout = 30
        result = self.test_performer.set_options(generator, testerClass, self.threads, self.num_values, self.update_results, processing_timeout=timeout)

        if not result is True:
            dlg = wx.MessageDialog(self.window, str(result),
                                   "Parameter Error", wx.OK|wx.ICON_EXCLAMATION)
            result = dlg.ShowModal()
            dlg.Destroy()
            return

        for i in range(self.threads):
            self.update_results(i, "[%d] Starting testing" % (i + 1), 0)
            self._results[i].SetValue("")

        self.startButton.Enable(False)
        self.stopButton.Enable()
        self.generatorList.Enable(False)
        self.testerList.Enable(False)
        self.threadList.Enable(False)
        self.valueList.Enable(False)
        self.progress.SetRange(self.num_values)
        self.progress.SetValue(0)

        self.test_performer.start_evaluation()


    def extract_class(self, control):
        val = control.GetValue()
        pos = string.find(val, '<')
        module = val[(pos + 1):-1]
        name = val[:(pos - 1)]
        name = name.strip()
        the_class = getattr(self.modules[module], name)
        return the_class

    def stop_test(self, event):
        if self.test_performer:
            self.test_performer.stop_evaluation()
            for i in range(self.threads):
                self.update_results(i, "[%d] Stopped testing" % (i + 1), 0)
            self.startButton.Enable()
            self.stopButton.Enable(False)
            self.generatorList.Enable()
            self.testerList.Enable()
            self.threadList.Enable()
            self.valueList.Enable()
            self.progress.SetValue(0)

    def on_param_update(self, event):
        thread, params, score, message, param_set = event.get_value()
        if message == "GENERATOR_FINISHED":
            for idx in range(self.threads):
                self._results[idx].AppendText(("[%d] Parameter generator finished, no more parameter sets can be generated" % (idx + 1)) + "\n")
        elif message == "BEST" or message == "FINISHED":
            testerCl = self.extract_class(self.testerList)
            tester = testerCl()
            config = "No results available"
            if params != None:
                config = tester.get_config(params)
                self.bestScore.SetValue("%f" % score)
            self.bestParams.SetValue(config)
            if message == "FINISHED":
                for idx in range(self.threads):
                    self._results[idx].AppendText(("[%d] Best result: %.5f" % (idx + 1, score)) + "\n")
                self.startButton.Enable()
                self.stopButton.Enable(False)
                self.progress.SetValue(self.progress.GetRange())
                self.generatorList.Enable()
                self.testerList.Enable()
                self.threadList.Enable()
                self.valueList.Enable()
        else:
            if len(self._results[thread].GetValue()) > self.buffer_limit:
                newval = self._results[thread].GetValue()
                newval = newval[-self.buffer_limit:]
                first_nl = newval.find("\n")
                self._results[thread].SetValue(newval[first_nl+1:])
            self._results[thread].AppendText(message + "\n")
            if param_set != 0 and param_set > self.progress.GetValue():
                self.progress.SetValue(param_set)

    def on_config(self, event):
        testerClass = self.extract_class(self.testerList)
        params = ParamGUI.ParamGUI(self.window, testerClass)

    ##############
    ## CALLBACK ##
    ##############
    def update_results(self, thread, result, param_set):
        if str(thread) == "BEST" or str(thread) == "FINISHED" or str(thread) == "GENERATOR_FINISHED":
            the_thread, params, score = result
            value = (the_thread, params, score, str(thread), param_set)
        else:
            value = (thread, {}, 0, result, param_set) 
        
        evt = UpdateEvent(myEVT_PARAM_UPDATE, -1, value)
        wx.PostEvent(self, evt)

    ################################
    ## COMBOBOX FILLING FUNCTIONS ##
    ################################
    def find_generators(self):
        dirs = os.listdir("generators")
        for filename in dirs:
            if filename[-3:] != ".py" or filename[:2] == "__":
                continue
            modulename = "generators." + filename[:-3]
            module = __import__(modulename)
            self.modules[modulename] = eval("module." + filename[:-3])
            if not hasattr(self.modules[modulename], "generators"):
                continue
            for idx in range(len(self.modules[modulename].generators)):
                name = self.modules[modulename].generators[idx]
                package = modulename
                self.generators.append((package, name))
        for package, name in self.generators:
            self.add_generator(name, package)

    def find_testers(self):
        dirs = os.listdir("testers")
        for filename in dirs:
            if filename[-3:] != ".py" or filename[:2] == "__":
                continue
            modulename = "testers." + filename[:-3]
            module = __import__(modulename)
            self.modules[modulename] = eval("module." + filename[:-3])
            if not hasattr(self.modules[modulename], "testers"):
                continue
            path = os.path.abspath("testers/" + filename)
            checksum = self.md5sum(path)
            for idx in range(len(self.modules[modulename].testers)):
                name = self.modules[modulename].testers[idx]
                package = modulename
                self.testers.append((package, name, checksum))
        for package, name, checksum in self.testers:
            self.add_tester(name, package)

    def md5sum(self, path):
        f = open(path, "rb")
        contents = f.read()
        f.close()
        m = md5.new()
        m.update(contents)
        return m.hexdigest()
        
    def add_generator(self, name, package):
        self.generatorList.Append("%s <%s>" % (name, package))

    def add_tester(self, name, package):
        self.testerList.Append("%s <%s>" % (name, package))

    ######################
    ## FILE I/O METHODS ##
    ######################
    def load_config(self):
        """
        This method loads the configuration from the user's homedir
        """
        filename = os.path.expanduser("~/.param_opt.conf")
        if not os.path.exists(filename):
            print "No configuration file found at %s" % filename
            return
        try:
            f = open(filename, "rb")
            pickler = cPickle.Unpickler(f)
            self.config = pickler.load()
            print "Succesfully loaded configuration from file %s" % filename
        except:
            self.config = []
            print "Loading data from configuration file %s failed" % filename

        if self.config:
            for package, name, checksum, settings in self.config:
                if package == "main" and name == "main":
                    self.generatorList.SetStringSelection(settings['generator'])
                    self.testerList.SetStringSelection(settings['tester'])
                    self.threadList.SetStringSelection(settings['threads'])
                    self.valueList.SetStringSelection(settings['values'])
                    self.timeout.SetValue(settings['timeout'])
                    self.update_threads(None)
                    self.update_values(None)
                    continue
                for p, n, c in self.testers:
                    if p != package or n != name:
                        continue
                    do_import = True
                    if c != checksum:
                        msg = "The module %s from package %s has changed since " \
                              "the configuration file has been saved. Do you " \
                              "want to load the stored configuration for this " \
                              "module anyway?" % (name, package)
                        dlg = wx.MessageDialog(self.window, str(msg), "Module changed", 
                                               wx.YES|wx.NO|wx.ICON_QUESTION)
                        result = dlg.ShowModal()
                        dlg.Destroy()
                        do_import = result == wx.ID_YES
                    if do_import:
                        the_class = getattr(self.modules[package], name)
                        the_class.param_names = settings['param_names']
                        the_class.parameters = settings['parameters']
                        the_class.config_file = settings['config_file']
                        the_class.start_command = settings['start_command']
                        the_class.param_path = settings['param_path']
                        the_class.max_reps = settings['max_reps']

    def write_config(self):
        """
        This method saves the configuration to the user's homedir
        """
        self.config = []
        for package, name, checksum in self.testers:
            the_class = getattr(self.modules[package], name)
            settings = {}
            settings['param_names'] = the_class.param_names
            settings['parameters'] = the_class.parameters
            settings['config_file'] = the_class.config_file
            settings['start_command'] = the_class.start_command
            settings['param_path'] = the_class.param_path
            settings['max_reps'] = the_class.max_reps
            self.config.append((package, name, checksum, settings))

        mainconf = {}
        mainconf['generator'] = self.generatorList.GetValue()
        mainconf['tester'] = self.testerList.GetValue()
        mainconf['threads'] = self.threadList.GetValue()
        mainconf['values'] = self.valueList.GetValue()
        mainconf['timeout'] = self.timeout.GetValue()
        self.config.append(("main", "main", "", mainconf))

        filename = os.path.expanduser("~/.param_opt.conf")
        try:
            f = open(filename, "wb")
            pickler = cPickle.Pickler(f, cPickle.HIGHEST_PROTOCOL)
            pickler.dump(self.config)
            print "Succesfully saved configuration to file %s" % filename
        except:
            print "Saving data to configuration file %s failed" % filename

if __name__ == "__main__":
    app = ParameterOptimizerGUI()
    app.MainLoop()
