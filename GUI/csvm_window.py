from Tkinter import *

import tkMessageBox
import Tkinter


#make a new window
root = Tkinter.Tk()
root.wm_title("CSVM Experiment Interface")
root.geometry("600x400")

lb1 = Listbox(root)
lb1.insert(1, "MNIST")
lb1.insert(2, "CIFAR10")
lb1.pack()
lb1.place(x = 0, y = 0)
root.mainloop()

