#!/bin/bash

echo "" > thrust_perf.txt
echo "" > sml_perf.txt

for n in {10000000..200000000..20000000};
do
	for i in `seq 1 10`;
	do
        ./thrust $n >> thrust_perf.txt
		./filter $n >> sml_perf.txt
	done 
done 
