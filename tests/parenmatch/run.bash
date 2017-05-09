#!/bin/bash


for n in {100000000..2000000000..100000000};
do
	for i in `seq 1 10`;
	do
		./parenmatch $n
	done 
done 
