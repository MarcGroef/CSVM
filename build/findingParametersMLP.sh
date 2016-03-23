#!/bin/bash


NHIDDENUNITS=150
#LEARNINGRATE=.1

line_number_hidden_units=$(awk '/nHiddenUnits/{ print NR; exit }' settings)
#line_number_learningRate=$(awk '/learningRate/{ print NR; exit }' settings)

#for i in {1..20}
#do
#	$LEARNINGRATE=`echo $LEARNINGRATE + 0.05 | bc`
	#((LEARNINGRATE=LEARNINGRATE+.05)) | bc 
#	echo $LEARNINGRATE
#	sed -i settings -e "$line_number_learningRate s/.*/learningRate $LEARNINGRATE/"
	for j in {1..100}
	do
		((NHIDDENUNITS=NHIDDENUNITS+20))
		sed -i settings -e "$line_number_hidden_units s/.*/nHiddenUnits $NHIDDENUNITS/"
		./CSVM settings
	done
#done

