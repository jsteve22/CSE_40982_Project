#!/usr/bin/env bash

for i in $(seq 1 500) 
do
	python main.py > "samples/$i.out"
	echo "done $i"
done
