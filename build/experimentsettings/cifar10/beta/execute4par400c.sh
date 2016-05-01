#!/bin/bash
cd ../../..
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results1 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results2 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results3 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results4 &
wait
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results5 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results6 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results7 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results8 &
wait
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results9 &
./CSVM experimentsettings/cifar10/beta/max/400c/settings > experimentsettings/cifar10/beta/max/400c/results10 &

./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results1 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results2 &

wait

./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results3 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results4 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results5 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results6 &

wait

./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results7 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results8 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results9 &
./CSVM experimentsettings/cifar10/beta/sum/400c/settings > experimentsettings/cifar10/beta/sum/400c/results10 &
