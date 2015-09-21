import wx
import string
import testers.SVMTester

class ParamGUI(wx.Frame):
    """
    ParamGUI is a configuration frame that can set up all the
    parameters, the configuration file and some other parameters
    related to the evaluating of an algorithm
    """
    def __init__(self, parent, tester):
        """
        Set up the required data members. The parent is the parent frame, the
        tester is the ParamTester class from the testers folder that is being
        set up
        """
        super(ParamGUI, self).__init__(parent, title="Parameter Settings", size=(800, 500))
        self.param_names = []
        self.param_settings = []
        self.tester = tester
        self._setup_gui()
        tester.add_parameters(self)
        self.changed = False
        self._setting_changed = False
        self._setting_idx = None
        self._update_list()
        self._type_int = False

        # Make sure the proper values are displayed for the first setting
        self.change_setting(None)
        self.Center()
        self.Show()
        self.MakeModal()

    def _update_list(self):
        """
        This updates the list with the parameters from the tester
        """
        for idx, name in enumerate(self.param_names):
            self.paramList.Append("%s" % name)
        self.paramList.Select(0)

    def _setup_gui(self):
        """
        This private method sets up the GUI
        """
        # Initialize frame
        self.Bind(wx.EVT_CLOSE, self.on_cancel)
        self.dialog_panel = wx.Panel(self)
        self.notebook = wx.Notebook(self.dialog_panel, id=wx.ID_ANY, style=wx.BK_DEFAULT)

        # Set up frame contents
        self._setup_parameter_panel()
        self._setup_configuration_panel()
        self.notebook.AddPage(self.first_panel, "Parameters")
        self.notebook.AddPage(self.second_panel, "Configuration file")
        
        # Set up Ok en Cancel buttons at the bottom
        self._setup_button_box()

        # Add everything to the panel and set up sizer
        self.dialog_sizer = wx.BoxSizer(wx.VERTICAL)
        self.dialog_sizer.Add(self.notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.dialog_sizer.Add(self.buttonBox, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.dialog_panel.SetSizer(self.dialog_sizer, True)

    def _setup_button_box(self):
        """
        This private method sets up the ok and cancel buttons of the dialog
        """
        self.okButton = wx.Button(self.dialog_panel, label='Ok')
        self.okButton.Bind(wx.EVT_BUTTON, self.on_ok)
        self.cancelButton = wx.Button(self.dialog_panel, label='Cancel')
        self.cancelButton.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.buttonBox = wx.BoxSizer()
        self.buttonBox.Add(self.okButton, proportion=1, flag=wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, border=5)
        self.buttonBox.Add(self.cancelButton, proportion=1, flag=wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, border=5)

    def _setup_parameter_panel(self):
        """
        This private method sets up the first panel, the one where the parameters of the
        algorithm are defined.
        """
        self.first_panel = wx.Panel(self.notebook)
        self._setup_param_overview()
        self._setup_param_settings()

        self.hbox = wx.BoxSizer()
        self.hbox.Add(self.vboxParams, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.hbox.Add(self.vboxSettings, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.first_panel.SetSizer(self.hbox, True)

    def _setup_param_overview(self):
        """
        This private method sets up the list of parameters and the add and remove buttons
        """
        self.vboxParams = wx.BoxSizer(wx.VERTICAL)
        self.paramLabel = wx.StaticText(self.first_panel, label='Parameters')
        self.paramList = wx.ListBox(self.first_panel)
        self.paramList.Bind(wx.EVT_LISTBOX, self.change_setting)

        self.newText = wx.TextCtrl(self.first_panel, value="")
        self.addButton = wx.Button(self.first_panel, label='Add')
        self.addButton.Bind(wx.EVT_BUTTON, self.on_add)
        self.delButton = wx.Button(self.first_panel, label='Delete')
        self.delButton.Bind(wx.EVT_BUTTON, self.on_delete)
        self.paramButtonBox = wx.BoxSizer()
        self.paramButtonBox.Add(self.newText, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)
        self.paramButtonBox.Add(self.addButton, proportion=0, flag=wx.ALL | wx.EXPAND, border=5)
        self.paramButtonBox.Add(self.delButton, proportion=0, flag=wx.ALL | wx.EXPAND, border=5)

        self.vboxParams.Add(self.paramLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxParams.Add(self.paramList, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxParams.Add(self.paramButtonBox, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)

    def _setup_param_settings(self):
        """
        This private method sets up controls of the settings for the currently selected parameter
        """
        self.vboxSettings = wx.BoxSizer(wx.VERTICAL)
        self.minLabel = wx.StaticText(self.first_panel, label='Minimum')
        self.minText = wx.TextCtrl(self.first_panel, value="0")
        self.maxLabel = wx.StaticText(self.first_panel, label='Maximum')
        self.maxText = wx.TextCtrl(self.first_panel, value="100")

        self.typeLabel = wx.StaticText(self.first_panel, label='Type')
        self.typeList = wx.ComboBox(self.first_panel, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.typeList.Append("Integer")
        self.typeList.Append("Float")
        self.typeList.Append("Static value")
        self.typeList.SetStringSelection("Float")

        self.scalingLabel = wx.StaticText(self.first_panel, label='Parameter scaling')
        self.scalingList = wx.ComboBox(self.first_panel, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.scalingList.Append("Linear")
        self.scalingList.Append("Logarithmic")
        self.scalingList.SetStringSelection("Linear")

        self.distributionLabel = wx.StaticText(self.first_panel, label='Distribution')
        self.distributionList = wx.ComboBox(self.first_panel, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.distributionList.Append("Uniform")
        self.distributionList.Append("Gaussian")
        self.distributionList.Bind(wx.EVT_COMBOBOX, self.update_distribution)
        self.typeList.Bind(wx.EVT_COMBOBOX, self.update_type)
        self.distributionList.SetStringSelection("Uniform")

        self.muLabel = wx.StaticText(self.first_panel, label='Mean')
        self.muText = wx.TextCtrl(self.first_panel, value="10")
        self.muText.Enable(False)
        self.sigmaLabel = wx.StaticText(self.first_panel, label='Standard deviation')
        self.sigmaText = wx.TextCtrl(self.first_panel, value="2")
        self.sigmaText.Enable(False)

        self.saveButton = wx.Button(self.first_panel, label='Save')
        self.saveButton.Bind(wx.EVT_BUTTON, self.on_save)

        self.minText.Bind(wx.EVT_CHAR, self.keypress_check)
        self.maxText.Bind(wx.EVT_CHAR, self.keypress_check)
        self.muText.Bind(wx.EVT_CHAR, self.keypress_float_check)
        self.sigmaText.Bind(wx.EVT_CHAR, self.keypress_float_check)

        self.vboxSettings.Add(self.minLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.minText, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.maxLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.maxText, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.typeLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.typeList, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.scalingLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.scalingList, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.distributionLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.distributionList, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.muLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.muText, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.sigmaLabel, proportion=0, flag=wx.LEFT, border=5)
        self.vboxSettings.Add(self.sigmaText, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.vboxSettings.Add(self.saveButton, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)

    def _setup_configuration_panel(self):
        """
        This method sets up the second panel, the one where the configuration file
        can be edited and the location of the algorithm can be set
        """
        self.second_panel = wx.Panel(self.notebook)

        self.config_path_label = wx.StaticText(self.second_panel, label='Configuration file path')
        self.config_path_text = wx.TextCtrl(self.second_panel)
        self.config_path_text.SetValue(self.tester.param_path)
        self.command_label = wx.StaticText(self.second_panel, label='Start command')
        self.command_text = wx.TextCtrl(self.second_panel)
        self.command_text.SetValue(self.tester.start_command)
        self.max_reps_label = wx.StaticText(self.second_panel, label='Maximum number of evaluations of a single parameter set')
        self.max_reps_text = wx.TextCtrl(self.second_panel)
        self.max_reps_text.SetValue(str(self.tester.max_reps))

        self.config_label = wx.StaticText(self.second_panel, label='Configuration file')
        self.config_editor = wx.TextCtrl(self.second_panel, style=wx.TE_MULTILINE | wx.HSCROLL)
        self.config_editor.SetValue(self.tester.config_file)

        self.config_path_text.Bind(wx.EVT_CHAR, self.keypress_config)
        self.command_text.Bind(wx.EVT_CHAR, self.keypress_config)
        self.max_reps_text.Bind(wx.EVT_CHAR, self.keypress_config)
        self.config_editor.Bind(wx.EVT_CHAR, self.keypress_config)

        font = self.config_editor.GetFont()
        font = wx.Font(font.GetPointSize(), wx.TELETYPE,
                       font.GetStyle(),
                       font.GetWeight(), font.GetUnderlined())
        self.config_editor.SetFont(font)

        self.config_sizer = wx.BoxSizer(wx.VERTICAL)
        self.config_sizer.Add(self.config_path_label, proportion=0)
        self.config_sizer.Add(self.config_path_text, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.config_sizer.Add(self.command_label, proportion=0)
        self.config_sizer.Add(self.command_text, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        self.config_sizer.Add(self.max_reps_label, proportion=0)
        self.config_sizer.Add(self.max_reps_text, proportion=0, flag=wx.ALL, border=5)
        self.config_sizer.Add(self.config_label, proportion=0)
        self.config_sizer.Add(self.config_editor, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.second_panel.SetSizer(self.config_sizer)

    ####################
    ## EVENT HANDLERS ##
    ####################
    def update_type(self, event):
        """
        This method is called when the Type button is changed or when other
        functions have changed the currently selected parameter. It changes
        the available/enabled settings for the selected parameter in the GUI.
        """
        self._setting_changed = True
        val = self.typeList.GetValue()
        if val == "Static value":
            self._type_int = False
            self.muText.Enable(False)
            self.sigmaText.Enable(False)
            self.maxText.Enable(False)
            self.distributionList.Enable(False)
            self.scalingList.Enable(False)
            self.minLabel.SetLabel("Value")
        else:
            self.maxText.Enable(True)
            self.distributionList.Enable(True)
            self.scalingList.Enable(True)
            self.update_distribution(None)
            self.minLabel.SetLabel("Minimum")
            if val == "Integer":
                if not self._type_int:
                    self.minText.SetValue(str(int(round(float(self.minText.GetValue())))))
                    self.maxText.SetValue(str(int(round(float(self.maxText.GetValue())))))
                    self._type_int = True
            elif val == "Float":
                if self._type_int:
                    self._type_int = False

    def update_distribution(self, event):
        """
        This method is called when the distribution is changed or when other
        functions have changed the currently selected parameter. It changes
        the available/enabled settings for the selected parameter in the GUI
        """
        self._setting_changed = True
        val = self.distributionList.GetValue()
        if val == "Uniform":
            self.muText.Enable(False)
            self.sigmaText.Enable(False)
        else:
            self.muText.Enable(True)
            self.sigmaText.Enable(True)

    def keypress_config(self, event):
        """
        This method is called when a key is pressed in a textbox that
        changes the current configuration. It sets the changed flag.
        TODO: check if the contents have actually changed
        """
        self.changed = True
        event.Skip()

    def keypress_other_check(self, event):
        """
        This method is called when a key is pressed in a textbox that
        changes the current setting. It sets the setting changed flag.
        TODO: check if the contents have actually changed
        """
        self._setting_changed = True
        event.Skip()

    def keypress_check(self, event):
        if self._type_int:
            self.keypress_integer_check(event)
        else:
            self.keypress_float_check(event)

    def keypress_integer_check(self, event):
        """
        This method is called when a key is pressed in a textbox
        that only accepts integers. If any non-integer is entered,
        it is ignored. The method also sets the changed flag if
        a proper character was pressed
        """
        self._setting_changed = True
        control = event.GetEventObject()
        keycode = event.GetKeyCode()
        if keycode >= 48 and keycode <= 57 or keycode < 32:
            event.Skip()
        elif keycode == 45 and control.GetInsertionPoint() == 0:
            event.Skip()

    def keypress_float_check(self, event):
        """
        This method is called when a key is pressed in a textbox
        that only accepts floats. If any non-float character is entered,
        it is ignored. The method also sets the changed flag if
        a proper character was pressed
        """
        self._setting_changed = True
        control = event.GetEventObject()
        keycode = event.GetKeyCode()
        if keycode >= 48 and keycode <= 57 or keycode < 32:
            event.Skip()
        elif keycode == 45 and control.GetInsertionPoint() == 0:
            event.Skip()
        elif keycode == 46:
            val = control.GetValue()
            if string.find(val, '.') == -1:
                event.Skip()

    def change_setting(self, event):
        """
        This method is called when a different setting is selected from
        the list, or when the value is changed programmatically, for example
        to force the save changed dialog to come up.
        """
        if self._setting_changed:
            dlg = wx.MessageDialog(self, "Do you want to save the changes you made to the current setting?",
                                   "Save Changes?", wx.YES|wx.NO|wx.ICON_QUESTION)
            result = dlg.ShowModal()
            dlg.Destroy()
            if result == wx.ID_YES:
                self.on_save(None)
        val = self.paramList.GetSelections()
        if len(val) == 0:
            self._setting_idx = None
            return
        idx = val[0]
        self._setting_idx = idx
        settings = self.param_settings[idx]
        if settings['type'] == "static":
            self.typeList.SetStringSelection("Static value")
            self.minText.SetValue(str(settings['value']))
            self.maxText.SetValue("")
            self.muText.SetValue("")
            self.sigmaText.SetValue("")
        else:
            self.minText.SetValue(str(settings['min']))
            self.maxText.SetValue(str(settings['max']))
            if settings['type'] == "int":
                self.typeList.SetStringSelection("Integer")
            else:
                self.typeList.SetStringSelection("Float")
            if settings['scaling'] == "log":
                self.scalingList.SetStringSelection("Logarithmic")
            else:
                self.scalingList.SetStringSelection("Linear")
            if settings['distribution'] == "gaussian":
                self.distributionList.SetStringSelection("Gaussian")
            else:
                self.distributionList.SetStringSelection("Uniform")
            if settings['distribution'] == "gaussian" and len(settings['value']) == 2:
                mu, sigma = settings['value']
                self.muText.SetValue(str(mu))
                self.sigmaText.SetValue(str(sigma))
            else:
                self.muText.SetValue("")
                self.sigmaText.SetValue("")

        self.update_type(None)
        self._setting_changed = False

    def on_save(self, event):
        """
        This method is called when the save button is pressed, or when the
        OK button is pressed, to make sure no changes are lost.
        """
        if self._setting_idx == None:
            return
        idx = self._setting_idx

        self._setting_changed = False
        settings = {}
        type = self.typeList.GetValue()
        if type == "Static value":
            settings['type'] = "static"
            try:
                settings['value'] = int(self.minText.GetValue())
            except:
                try:
                    settings['value'] = float(self.minText.GetValue())
                except:
                    settings['value'] = self.minText.GetValue()
        else:
            if type == "Integer":
                settings['type'] = "int"
                settings['min'] = int(round(float(self.minText.GetValue())))
                settings['max'] = int(round(float(self.maxText.GetValue())))
            else:
                settings['type'] = "float"
                settings['min'] = float(self.minText.GetValue())
                settings['max'] = float(self.maxText.GetValue())
            scaling = self.scalingList.GetValue()
            if scaling == "Linear":
                settings['scaling'] = "linear"
            else:
                settings['scaling'] = "log"
                if abs(settings['min']) < 0.0000001 and settings['min'] >= 0:
                    settings['min'] = 0.0000001
                elif abs(settings['min']) < 0.0000001 and settings['min'] <= 0:
                    settings['min'] = -0.0000001

            distribution = self.distributionList.GetValue()
            if distribution == "Uniform":
                settings['distribution'] = "uniform"
            else:
                settings['distribution'] = "gaussian"
                try:
                    mu = float(self.muText.GetValue())
                    sigma = float(self.sigmaText.GetValue())
                except:
                    mu = 0
                    sigma = 1
                settings['value'] = (mu, sigma)

        if settings != self.param_settings[idx]:
            self.changed = True
            self.param_settings[idx] = settings

    def on_ok(self, event):
        """
        This method is called when the OK button is pressed. It saves the
        changes and closes the window
        """
        if self._setting_changed:
            self.change_setting(None)

        self.tester.parameters = {}
        for idx, name in enumerate(self.param_names):
            self.tester.set_parameter(name, self.param_settings[idx])
        self.tester.param_names = self.param_names
        print repr(self.tester.parameters)
        self.tester.config_file = self.config_editor.GetValue()
        self.tester.start_command = self.command_text.GetValue()
        self.tester.param_path = self.config_path_text.GetValue()
        try:
            self.tester.max_reps = int(self.max_reps_text.GetValue())
        except ValueError:
            pass
        self._setting_changed = False
        self.MakeModal(False)
        self.Destroy()

    def on_cancel(self, event):
        """
        This method is called when the Cancel button is pressed. It cancels
        the changes and closes the window
        """
        if not self.changed and not self._setting_changed:
            self.MakeModal(False)
            self.Destroy()
            return

        dlg = wx.MessageDialog(self, "Do you really want to abandon the changes?",
                               "Confirm Cancel", wx.YES|wx.NO|wx.ICON_QUESTION)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_YES:
            self.MakeModal(False)
            self.Destroy()

    def on_add(self, event):
        """
        This method is called when the Add button is pressed. It adds a new
        setting to the parameterlist.
        """
        name = self.newText.GetValue()
        if name == "":
            return

        self.newText.SetValue("")
        settings = {"type": "static", "value": 0}
        self.param_names.append(name)
        self.param_settings.append(settings)
        self.changed = True
        self.paramList.Append(name)
        self.paramList.Select(len(self.param_names) - 1)
        self.change_setting(None)

    def on_delete(self, event):
        """
        This method is called when the Delete button is pressed. It deletes
        the currently selected setting from the list
        """
        val = self.paramList.GetSelections()
        if len(val) == 0:
            self._setting_idx = None
            return
        idx = val[0]
        del self.param_names[idx]
        del self.param_settings[idx]
        self.paramList.Delete(idx)
        self.changed = True

    ########################
    ## GENERATOR INTERFACE #
    ########################
    # Provided to be able to use the add_parameters method from the
    # parameter tester
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

    # Same as above
    def set_max_reps(self, max_reps):
        pass

if __name__ == "__main__":
    from ParamOpt import ParameterOptimizerGUI
    app = ParameterOptimizerGUI()
    params = ParamGUI(app.window, testers.SVMTester.SVMTester)
    app.MainLoop()

