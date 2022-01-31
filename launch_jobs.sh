#!/bin/bash

N=3

for i in {1..3}
do
    echo "Starting job number $i out of $N"
    qsub job
done
