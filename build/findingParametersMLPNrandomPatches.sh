#!/bin/bash


NRANDOMPATCHES=30000
#LEARNINGRATE=.1

line_number_hidden_units=$(awk '/nRandomPatches/{ print NR; exit }' settings)
#line_number_learningRate=$(awk '/learningRate/{ print NR; exit }' settings)

#for i in {1..20}
#do
#	$LEARNINGRATE=`echo $LEARNINGRATE + 0.05 | bc`
	#((LEARNINGRATE=LEARNINGRATE+.05)) | bc 
#	echo $LEARNINGRATE
#	sed -i settings -e "$line_number_learningRate s/.*/learningRate $LEARNINGRATE/"
	for j in {1..5}
	do
		((NRANDOMPATCHES=NRANDOMPATCHES+20000))
		sed -i settings -e "$line_number_hidden_units s/.*/nRandomPatches $NRANDOMPATCHES/"
		./CSVM settings
	done
#done

