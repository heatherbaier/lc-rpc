#!/bin/bash

N=1

for i in {1..1}
do
    echo "Starting job number $i out of $N"
    qsub job
done
