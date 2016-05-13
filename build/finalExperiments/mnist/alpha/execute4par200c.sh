#!/bin/bash
cd ../../..
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results1 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results2 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results3 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results4 &
wait
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results5 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results6 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results7 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results8 &
wait
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results9 &
./CSVM finalExperiments/mnist/alpha/max/200c/settings > finalExperiments/mnist/alpha/max/200c/results10 &

./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results1 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results2 &

wait

./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results3 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results4 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results5 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results6 &

wait

./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results7 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results8 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results9 &
./CSVM finalExperiments/mnist/alpha/sum/200c/settings > finalExperiments/mnist/alpha/sum/200c/results10 &
