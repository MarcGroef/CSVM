#!/bin/bash
cd ../../..
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results1 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results2 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results3 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results4 &
wait
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results5 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results6 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results7 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results8 &
wait
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results9 &
./CSVM finalExperiments/mnist/gamma/max/800c/settings > finalExperiments/mnist/gamma/max/800c/results10 &

./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results1 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results2 &

wait

./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results3 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results4 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results5 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results6 &

wait

./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results7 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results8 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results9 &
./CSVM finalExperiments/mnist/gamma/sum/800c/settings > finalExperiments/mnist/gamma/sum/800c/results10 &
