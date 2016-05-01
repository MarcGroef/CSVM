#!/bin/bash
cd ../../../..
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results1 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results2 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results3 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results4 &
wait
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results5 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results6 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results7 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results8 &
wait
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results9 &
./CSVM experimentsettings/mnist/beta/max/800c/settings > experimentsettings/mnist/beta/max/800c/results10 &

./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results1 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results2 &

wait

./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results3 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results4 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results5 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results6 &

wait

./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results7 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results8 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results9 &
./CSVM experimentsettings/mnist/beta/sum/800c/settings > experimentsettings/mnist/beta/sum/800c/results10 &
