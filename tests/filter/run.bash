#!/bin/bash
echo "" > sml_perf.txt

for n in {100000000..2000000000..200000000};
do
	for i in `seq 1 10`;
	do
		./filter $n >> sml_perf.txt
	done 
done 
