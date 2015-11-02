#!/bin/bash

mv generateCodebook._py generateCodebook.py
python generateCodebook.py
mv generateCodebook.py generateCodebook._py
