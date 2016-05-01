#!/bin/bash
cd ../../../..
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results1 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results2 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results3 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results4 &
wait
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results5 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results6 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results7 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results8 &
wait
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results9 &
./CSVM experimentsettings/mnist/beta/max/200c/settings > experimentsettings/mnist/beta/max/200c/results10 &

./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results1 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results2 &

wait

./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results3 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results4 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results5 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results6 &

wait

./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results7 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results8 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results9 &
./CSVM experimentsettings/mnist/beta/sum/200c/settings > experimentsettings/mnist/beta/sum/200c/results10 &
