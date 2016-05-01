#!/bin/bash
cd ../../..
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results1 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results2 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results3 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results4 &
wait
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results5 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results6 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results7 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results8 &
wait
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results9 &
./CSVM experimentsettings/cifar10/gamma/max/800c/settings > experimentsettings/cifar10/gamma/max/800c/results10 &

./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results1 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results2 &

wait

./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results3 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results4 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results5 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results6 &

wait

./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results7 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results8 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results9 &
./CSVM experimentsettings/cifar10/gamma/sum/800c/settings > experimentsettings/cifar10/gamma/sum/800c/results10 &
