#!/bin/bash
rm all.png
montage *.png -quality 100 -geometry +1+1 all.png
rm centr*
