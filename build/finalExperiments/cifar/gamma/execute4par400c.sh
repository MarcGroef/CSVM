#!/bin/bash
cd ../../..
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results1 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results2 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results3 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results4 &
wait
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results5 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results6 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results7 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results8 &
wait
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results9 &
./CSVM finalExperiments/cifar/gamma/max/400c/settings > finalExperiments/cifar/gamma/max/400c/results10 &

./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results1 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results2 &

wait

./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results3 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results4 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results5 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results6 &

wait

./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results7 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results8 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results9 &
./CSVM finalExperiments/cifar/gamma/sum/400c/settings > finalExperiments/cifar/gamma/sum/400c/results10 &
