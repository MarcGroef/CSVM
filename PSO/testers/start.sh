#!/bin/bash

CURDIR=`pwd`
PYTHONPATH=$PYTHONPATH:$CURDIR/.. python bandit.py
#YTHONPATH=$PYTHONPATH:$CURDIR/.. python NumPerTester.py
