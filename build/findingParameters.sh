#!/bin/bash


NHIDDENUNITS=100
VOTING="SUM"
#LEARNINGRATE=.1

line_number_hidden_units=$(awk '/nHiddenUnits/{ print NR; exit }' settings)
line_number_random_patches=$(awk '/nRandomPatches/{ print NR; exit }' settings)
line_voting=$(awk '/voting/{ print NR; exit }' settings)

#line_number_learningRate=$(awk '/learningRate/{ print NR; exit }' settings)

#for i in {1..20}
#do
#	$LEARNINGRATE=`echo $LEARNINGRATE + 0.05 | bc`
	#((LEARNINGRATE=LEARNINGRATE+.05)) | bc 
#	echo $LEARNINGRATE
#	sed -i settings -e "$line_number_learningRate s/.*/learningRate $LEARNINGRATE/"
	for j in {1..4}
		do
		sed -i settings -e "$line_number_hidden_units s/.*/nHiddenUnits $NHIDDENUNITS/"
		((NHIDDENUNITS=NHIDDENUNITS+100))
		
		VOTING="SUM"
		sed -i settings -e "$line_voting s/.*/voting $VOTING/"
			./CSVM settings	
			
		VOTING="MAJORITY"
		sed -i settings -e "$line_voting s/.*/voting $VOTING/"
			
		./CSVM settings			
	done
#done

