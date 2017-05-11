#!/bin/bash

echo "" > thrust_perf_scan.txt
echo "" > sml_perf_scan.txt

for n in {100000000..2000000000..200000000};
do
	for i in `seq 1 10`;
	do
        ./thrust $n >> thrust_perf_scan.txt
		./scan $n >> sml_perf_scan.txt
	done 
done 

mv *_scan.txt ~
