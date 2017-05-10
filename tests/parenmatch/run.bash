#!/bin/bash

echo "" > thrust_perf.txt
echo "" > sml_perf.txt

for n in {100000000..2000000000..100000000};
do
	for i in `seq 1 10`;
	do
		./thrust $n >> thrust_perf.txt
		./parendist $n >> sml_perf.txt
	done 
done 
