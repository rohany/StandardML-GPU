#!/bin/bash

echo "" > thrust_perf_mcss.txt
echo "" > sml_perf_mcss.txt

for n in {100000000..1000000000..200000000};
do
	for i in `seq 1 5`;
	do
        ./thrust $n >> thrust_perf_mcss.txt
		./mcss $n >> sml_perf_mcss.txt
	done 
done 

mv *_mcss.txt ~
