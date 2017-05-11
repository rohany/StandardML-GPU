#!/bin/bash

echo "" > thrust_perf_reduce.txt
echo "" > sml_perf_reduce.txt

for n in {10000000..200000000..20000000};
do
	for i in `seq 1 10`;
	do
        ./thrust $n >> thrust_perf.txt
		./reduce $n >> sml_perf.txt
	done 
done 

mv *_reduce.txt ~
