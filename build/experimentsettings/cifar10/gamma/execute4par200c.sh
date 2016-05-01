#!/bin/bash
cd ../../../..
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results1 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results2 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results3 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results4 &
wait
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results5 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results6 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results7 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results8 &
wait
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results9 &
./CSVM experimentsettings/cifar10/gamma/max/200c/settings > experimentsettings/cifar10/gamma/max/200c/results10 &

./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results1 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results2 &

wait

./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results3 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results4 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results5 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results6 &

wait

./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results7 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results8 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results9 &
./CSVM experimentsettings/cifar10/gamma/sum/200c/settings > experimentsettings/cifar10/gamma/sum/200c/results10 &
