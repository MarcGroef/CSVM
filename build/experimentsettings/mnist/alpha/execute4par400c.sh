#!/bin/bash
cd ../../../..
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results1 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results2 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results3 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results4 &
wait
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results5 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results6 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results7 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results8 &
wait
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results9 &
./CSVM experimentsettings/mnist/alpha/max/400c/settings > experimentsettings/mnist/alpha/max/400c/results10 &

./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results1 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results2 &

wait

./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results3 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results4 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results5 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results6 &

wait

./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results7 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results8 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results9 &
./CSVM experimentsettings/mnist/alpha/sum/400c/settings > experimentsettings/mnist/alpha/sum/400c/results10 &
